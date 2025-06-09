import os
import time
import json
import torch
import wandb

import torch.nn as nn
from tqdm import tqdm
from loguru import logger
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import transformers
transformers.logging.set_verbosity_error()

from utils import *

def main(args):
    set_seed(args)
    
    ############ Setup DDP environment ############
    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if global_rank != 0: logger.remove() # turn off logger

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    ############ Initialize wandb without config (it is passed later) ############
    if (not args.unset_wandb) and global_rank == 0:
        wandb.init(project=args.project, name=args.name, entity=args.entity)
    ############ Debug Setting ############
    if args.debug:
        p_prev = {}
        tracked_layers = ['module.model.layers.0.self_attn.q_proj.weight', 
                        'module.model.layers.0.mlp.down_proj.weight',
                        'module.model.layers.4.self_attn.q_proj.weight', 
                        'module.model.layers.4.mlp.down_proj.weight',
                        'module.model.layers.7.self_attn.q_proj.weight', 
                        'module.model.layers.7.mlp.down_proj.weight']

    ############ Setup training data ############
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    dataloader, tokenizer = setup_dataset(args, global_rank, world_size)

    ############ Initialize model ############
    model_config, model = setup_model(args)
    model.generation_config.pad_token_id = tokenizer.pad_token_id


    ############ Resuming from checkpoints ############
    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    # identifying checkpointing
    if args.continue_from is not None and os.path.exists(args.continue_from):
        # searching the latest checkpoints
        checkpoint_path_list = os.listdir(args.continue_from)
        checkpoint_path_list = [int(x.split('_')[-1]) for x in checkpoint_path_list if x.startswith('model_')]
        if len(checkpoint_path_list) > 0:
            logger.info('Find Checkpoints', checkpoint_path_list)
            beginning_step = max(checkpoint_path_list)
            if args.resume_step is not None:
                beginning_step = args.resume_step
            args.continue_from = os.path.join(args.continue_from, f"model_{beginning_step}")
            logger.info('Continue from', args.continue_from)
        else:
            logger.warning(f"Did not find any checkpoints in {args.continue_from}")
            args.continue_from = None

    # resuming from checkpointing
    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        load_model_weight(model, checkpoint_path, args)
        logger.info(f"Model successfully loaded (strict=False policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)


    ############ Setup model ############
    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(dtype=torch.bfloat16)
    model = model.to(device=device)

    for _, module in model.named_modules():
        if isinstance(module, QScaleLinear):
            weight_device = module.weight.device
            module.weight.scales = module.weight.scales.to(device=weight_device)
            module.weight.zeros = module.weight.zeros.to(device=weight_device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params_int8 = [p for p in model.parameters() if hasattr(p, 'group_size')]

    ############ Initialize wandb ############
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        if not args.unset_wandb:
            wandb.config.update(run_config, allow_val_change=True)
            wandb.save(os.path.abspath(__file__), policy="now") # save current script
        if args.set_tensorboard:
            writer = SummaryWriter(log_dir=args.tensorboard_dir)
            # writer = add_hparams(run_config, {})
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    ############ Initialize optimization ############
    if 'galore' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not (isinstance(module, nn.Linear) or isinstance(module, QScaleLinear) or isinstance(module, QLinear)): continue
            if not any(target_key in module_name for target_key in target_modules_list): continue
            galore_params.append(module.weight)

        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type,
                        "quant": args.proj_quant,'quant_n_bit': args.proj_bits, 'quant_group_size': args.proj_group_size,
                        'cos_threshold': args.cos_threshold, 'gamma_proj': args.gamma_proj, 'queue_size': args.queue_size}]
    elif 'spam' in args.optimizer.lower():
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not (isinstance(module, nn.Linear) or isinstance(module, QScaleLinear) or isinstance(module, QLinear)): continue
            if not any(target_key in module_name for target_key in target_modules_list): continue
            galore_params.append(module.weight)

        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]        
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'density': args.density, 'update_proj_gap': args.update_proj_gap}]
    else:
        param_groups = None
        id_galore_params = None

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")

    if args.simulation:
        num_train_params = sum(p.numel() for p in trainable_params)
    else:
        num_train_params = sum(p.numel() for p in trainable_params) + sum(p.numel() for p in trainable_params_int8)

    logger.info(f"Trainable params: {num_train_params / 1_000_000:.2f}M")
    if 'q_galore' in args.optimizer.lower():
        logger.info(f"Trainable params with Q-GaLore enabled: {sum(p.numel() for p in trainable_params_int8) / 1_000_000:.2f}M")
    elif 'galore' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    elif 'spam' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")

    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    model, optimizer, scheduler, layer_wise_flag = setup_optimization(args, model, trainable_params, param_groups, id_galore_params,model_config)

    # set model DDP
    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # resume optimizer
    if args.restore_optimizer and args.continue_from is not None:
        logger.info("Restoring optimizer and scheduler from the checkpoint")
        _optimizer_dir = args.continue_from
        optimizer_checkpoint = torch.load(os.path.join(_optimizer_dir, "optimizer.pt"), map_location="cpu")
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        update_step = optimizer_checkpoint["update_step"]
        beginning_step = update_step
        global_step = optimizer_checkpoint["global_step"]
        logger.info(f"Optimizer and scheduler restored from {_optimizer_dir}")
    # ##############################
    # HOOK 
    # ##############################
    activation = {}
    def make_hook(name):
        def hook(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            activation[name] = tensor.detach()
        return hook

    sqnr_acts = {}
    sqnr_weights = {}
    def sqnr_hook(name):
        def hook(module, input, output):
            assert isinstance(output, tuple)
            sqnr_act = output[1].items()
            sqnr_acts[name] = sqnr_act.detach()
            sqnr_weight = output[2].items()
            sqnr_weights[name] = sqnr_weight.detach()
    if args.debug:
        for idx, layer in enumerate(model.module.model.layers):
                layer.register_forward_hook(make_hook(f"layer_{idx}"))
        # if args.act_quant and args.weight_quant:
        #     for module_name, module in model.named_modules():
        #         if isinstance(module, QLinear):
        #             module.register_forward_hook(sqnr_hook(f"sqnr_{module_name}"))

        
    
    

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################
    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    total_svd_count = 0
    # model = torch.compile(model)

    for batch_idx, batch in enumerate(dataloader):

        if update_step != 0 and batch_idx <= args.gradient_accumulation * update_step: continue # skipping learned data when resuming from checkpointing

        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            logger.info(f"Rank {global_rank} stopping training.")
            break

        # forward & backward
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss

        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0: continue
        # -----------------------------------------------------------------------------
        # <-- ADDED CODE FOR GRADIENT NORM LOGGING (before clipping) -->
        # -----------------------------------------------------------------------------
        if not args.single_gpu:
            param_iterator = model.module.parameters()
        else:
            param_iterator = model.parameters()

        grad_l2_sum = torch.tensor(0.0, device=device)
        num_params_local = 0
        for p in param_iterator:
            if p.grad is not None:
                grad_l2_sum += p.grad.data.norm(2).pow(2)
                num_params_local+=1

        # dist.all_reduce(grad_l2_sum, op=dist.ReduceOp.SUM)  # gather sum of squares from all ranks
        global_grad_norm = grad_l2_sum.sqrt()
        # global_grad_norm = grad_l2_sum
    # -----------------------------------------        

        # The below code is only executed during the update step
        # add grad clipping: TODO: add gradient clipping of int8 weight
        if args.grad_clipping != 0.0:
            if args.grad_clipping==1.0: 
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            elif args.grad_clipping==1e-3:
                torch.nn.utils.clip_grad_value_(trainable_params, args.grad_clipping)
            else:
                assert False, "Wrong clipping Value"
        if global_rank == 0: pbar.update(1)
        global_grad_norm_after=torch.tensor(0.0)

        # ##############################
        #  SAVE PER-LAYER ACTIVATION
        # ##############################
        if args.debug and global_rank == 0:
            if update_step % args.eval_every_debug == 0:
                if args.fp4 or (args.act_quant and args.weight_quant) or args.quest:
                    for module_name, module in model.named_modules():
                        if isinstance(module, base_linear.QLinear) or isinstance(module, base_linear.Qfp4Linear) or isinstance(module, base_linear.QuantizedLinear):
                            sqnr_act = module.sqnr_act.item()
                            sqnr_acts[module_name] = sqnr_act
                            sqnr_weight = module.sqnr_weight.item()
                            sqnr_weights[module_name] = sqnr_weight
                    if args.set_tensorboard:
                        writer.add_scalars(f"sqnr_weights",sqnr_weights,update_step)
                        writer.add_scalars(f"sqnr_activations",sqnr_acts,update_step)
                    if not args.unset_wandb:
                        wandb.log(
                            {
                                "sqnr_weights":     sqnr_weights,   #= writer.add_scalars
                                "sqnr_activations": sqnr_acts
                            },
                            step=update_step
                        )

                var_dict = {}
                max_dict = {}
                for name, act in activation.items():
                    flat = act.flatten(start_dim=1) # flatten to [B, -1]
                    var = flat.var(dim=1, unbiased=False).mean().item()
                    var_dict[name] = var
                    mx = flat.abs().max().item()
                    max_dict[name] = mx
                if args.set_tensorboard:
                    writer.add_scalars(f"output_variance",var_dict,update_step)
                    writer.add_scalars(f"output_absmax",max_dict,update_step)
                if not args.unset_wandb:
                    wandb.log(
                        {
                            "output_variance": var_dict,            # 对应 writer.add_scalars
                            "output_absmax":   max_dict
                        },
                        step=update_step
                    )

            

        ################################
        
        if not layer_wise_flag: # layer-wise updation is done during backward; requires gradient_accumulation equals 1
            if args.optimizer.lower()=='spam':
                _,global_grad_norm_after=optimizer.step()
            elif args.optimizer.lower()=='stablespam':
                _,global_grad_norm_after=optimizer.step()
            elif args.optimizer.lower()=='stablespam8bit':
                _,global_grad_norm_after=optimizer.step()
            elif args.optimizer.lower()=='stablespamfp8':
                _,global_grad_norm_after=optimizer.step()
            elif args.optimizer.lower()=='stablespamfp4':
                _,global_grad_norm_after=optimizer.step()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time
        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            model.module.save_pretrained(current_model_directory, max_shard_size='100GB', from_pt=True)
            saving_model_weight(model.module, f"{current_model_directory}/pytorch_model.bin", args)

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir if not args.unset_wandb else None,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            if not args.unset_wandb:
                wandb_info = {
                    "wandb_id": wandb.run.id,
                }
                with open(f"{args.save_dir}/wandb.json", "w") as f:
                    json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            if args.dynamic_sqnr:
                if update_step % args.eval_every_sqnr == 0 and global_rank == 0:
                    from utils.sqnr_analysis import eval_sqnr_vs_loss
                    sqnr_res = eval_sqnr_vs_loss(
                        fp_model=model,
                        val_loader=val_dataloader,
                        criterion=loss_fn,
                        device="cuda",
                        bits=args.sqnr_bits,       # 例如 [4,6,8]
                        groups=args.sqnr_groups,   # 例如 [32,64,128]
                        max_batches=args.sqnr_batches
                    )

                    for (bit, grp), d in sqnr_res.items():
                        tag_prefix = f"sqnr_eval/bit{bit}_g{grp}"
                        if args.set_tensorboard:
                            writer.add_scalar(f"{tag_prefix}/loss",  d["loss"],  update_step)
                            writer.add_scalar(f"{tag_prefix}/sqnr",  d["sqnr_overall"], update_step)
                        if not args.unset_wandb:
                            wandb.log({
                                f"{tag_prefix}/loss": d["loss"],
                                f"{tag_prefix}/sqnr": d["sqnr_overall"]
                                },
                                step=update_step,
                            )
                        
                        # 可选：逐层
                        for ln, v in d["sqnr_per_layer"].items():
                            if args.set_tensorboard:
                                writer.add_scalar(f"{tag_prefix}/layer_{ln}", v, update_step)
                            if not args.unset_wandb:
                                wandb.log({
                                    f"{tag_prefix}/layer_{ln}": v,
                                    },
                                    step=update_step,
                                )

            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, perplexity,evaluated_on_tokens = evaluate_model(
                model, tokenizer, pad_idx, global_rank, world_size, device, args
            )
            if global_rank == 0:
                if not args.unset_wandb:
                    wandb.log({
                        "final_eval_loss": total_loss,
                        "final_eval_tokens": evaluated_on_tokens,
                        },
                        step=update_step,
                    )
                if args.set_tensorboard:
                    writer.add_scalar("final_eval_loss", total_loss, update_step)
                    writer.add_scalar("final_eval_tokens", evaluated_on_tokens, update_step)
                
            logger.info(f"Eval loss at step {update_step}: {total_loss}")
            logger.info(f"Eval perplexity at step {update_step}: {perplexity}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size
        total_svd_count = getting_svd_cnt(optimizer)

        if global_rank == 0:
            if not args.unset_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "total_svd_count": total_svd_count,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "grad_norm": global_grad_norm.item(),
                    "grad_norm_afterclip":global_grad_norm_after.item(),
                    "throughput_batches": batches_in_update / update_time,
                    },
                    step=update_step,
                )
            if args.set_tensorboard:
                writer.add_scalar("loss", loss.item(),update_step)
                writer.add_scalar("lr", lr,update_step)
                writer.add_scalar("tokens_seen", tokens_seen,update_step)
                writer.add_scalar("total_svd_count", total_svd_count,update_step)
                writer.add_scalar("throughput_tokens", tokens_in_update / update_time,update_step)
                writer.add_scalar("throughput_examples", args.total_batch_size / update_time,update_step)
                writer.add_scalar("grad_norm", global_grad_norm.item(),update_step)
                writer.add_scalar("grad_norm_afterclip",global_grad_norm_after.item(),update_step)
                writer.add_scalar("throughput_batches", batches_in_update / update_time,update_step)
                #plot  histogram for optimizer state
            if args.debug:
                if update_step % args.eval_every_debug == 0:
                    if update_step > args.eval_every_debug:
                        for name, p in model.named_parameters():
                            if name in tracked_layers and name in p_prev:
                                update_now = (p.data - p_prev[name]).view(-1)
                                update_prev = (p_prev[name] - p_prev_prev[name]).view(-1) if name in p_prev_prev else update_now
                                cos_sim = F.cosine_similarity(update_now, update_prev,dim = 0, eps = 1e-8)
                                if args.set_tensorboard:
                                    writer.add_scalar(f"cos_sim_update/{name}", cos_sim.item(),update_step)
                                if not args.unset_wandb:
                                    wandb.log({f"cos_sim_update/{name}": cos_sim.item()}, step=update_step)

                                
                    
                    p_prev_prev = p_prev
                    p_prev = {name:p.data.detach().clone() for name,p in model.named_parameters() if name in tracked_layers}
                    histograms = {}
                    for name, param in model.named_parameters():
                        if param in optimizer.state:
                            state = optimizer.state[param]
                            opt = args.optimizer.lower()
                            def _add_hist(key, tensor):
                                histograms[f"/opt/{key}/{name}"] = wandb.Histogram(tensor.float().cpu())
                            if opt in {"adam", "stablespam", "stablespamfp8"}:
                                if "exp_avg" in state:
                                    if not args.unset_wandb:
                                        if "exp_avg"   in state: _add_hist("exp_avg",   state["exp_avg"])
                                        if "exp_avg_sq" in state: _add_hist("exp_avg_sq", state["exp_avg_sq"])
                                    if args.set_tensorboard:
                                        if "exp_avg"   in state: writer.add_histogram(f"/opt/exp_avg/{name}", state["exp_avg"].to(dtype=torch.float32), update_step)
                                        if "exp_avg_sq" in state: writer.add_histogram(f"/opt/exp_avg_sq/{name}", state["exp_avg_sq"].to(dtype=torch.float32), update_step)
                            elif opt in {"adam8bit", "stablespam8bit"}:
                                if not args.unset_wandb:
                                    if "state1" in state: _add_hist("exp_avg",   state["state1"])
                                    if "state2" in state: _add_hist("exp_avg_sq", state["state2"])
                                if args.set_tensorboard:
                                        if "state1" in state: writer.add_histogram(f"/opt/exp_avg/{name}", state['state1'], update_step)
                                        if "state2" in state: writer.add_histogram(f"/opt/exp_avg_sq/{name}", state['state2'], update_step)
                    if opt in {"stablespam8bit", "adam8bit", "adamfp8", "stablespamfp8", "stablespamfp4"}:
                        sqnr_m_dict = {}
                        sqnr_v_dict = {}
                        for name, param in model.named_parameters():
                            if param in optimizer.state:
                                state = optimizer.state[param]
                                if state['sqnr_m'].item()>0 and 'proj' in name:
                                    sqnr_m_dict[name] = state['sqnr_m'].item()
                                    sqnr_v_dict[name] = state['sqnr_v'].item()
                        if args.set_tensorboard:
                            writer.add_scalars(f"sqnr_m", sqnr_m_dict, update_step)
                            writer.add_scalars(f"sqnr_v", sqnr_v_dict, update_step)
                        if not args.unset_wandb:
                            wandb.log({"sqnr_m": sqnr_m_dict, "sqnr_v": sqnr_v_dict}, step=update_step)
        update_time = time.time()
        # if update_step>1000:
        #     break

    # ##########  ####################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory, max_shard_size='100GB', from_pt=True)
        saving_model_weight(model.module, f"{current_model_directory}/pytorch_model.bin", args)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir if not args.unset_wandb else None,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, perplexity,evaluated_on_tokens = evaluate_model(
        model, tokenizer, pad_idx, global_rank, world_size, device, args
    )

    if global_rank == 0:
        if not args.unset_wandb:
            wandb.log({
                "final_eval_loss": total_loss,
                "final_eval_tokens": evaluated_on_tokens,
                },
                step=update_step,
            )
        if args.set_tensorboard:
            writer.add_scalar("final_eval_loss", total_loss, update_step)
            writer.add_scalar("final_eval_tokens", evaluated_on_tokens, update_step)
            writer.close()

        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
