import os
import torch
import argparse
from loguru import logger
from datetime import datetime
from .base_linear import QUANTIZER_CLASSES
import json

def parse_args(args):
    parser = argparse.ArgumentParser()


    ### Experiment setup ###
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--resume_step", type=int, default=None) # if None, resume from the latest checkpoint
    parser.add_argument("--restore_optimizer", action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)

    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    ### Debug setup ###
    parser.add_argument("--debug", default=False,action="store_true")
    parser.add_argument("--eval_every_debug",type=int, default=50)

    ### Training hyperparameters ###
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                            "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                            "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    parser.add_argument("--activation_checkpointing", action="store_true")


    ### Wandb ###
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--project", type=str, default="test")
    parser.add_argument("--unset_wandb", action="store_true")
    parser.add_argument("--entity", type=str, default=None)

    ##sqnr
    parser.add_argument("--dynamic_sqnr", action="store_true")
    parser.add_argument("--eval_every_sqnr", type=int, default=200)
    parser.add_argument("--sqnr_bits",       nargs="+", type=int, default=[4,6,8])
    parser.add_argument("--sqnr_groups",     nargs="+", type=int, default=[32,64,128])
    parser.add_argument("--sqnr_batches",    type=int, default=50)


    ### Adaptive optimization hyperparameters ###
    parser.add_argument("--beta1", type=float, default=0.9) # beta1 for Adafactor, GaLore_adafactor, (Q-)GaLore-adam or SGD
    parser.add_argument("--beta2", type=float, default=0.999) # beta2 for (Q-)GaLore-adam
    # GaLore hyperparameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=500)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    # Q-GaLore hyperparameters: quantization
    parser.add_argument("--proj_quant", action='store_true')
    parser.add_argument("--proj_bits", type=int, default=8)
    parser.add_argument("--proj_group_size", type=int, default=256)
    parser.add_argument("--weight_quant", action='store_true')
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--weight_group_size", type=int, default=256)
    parser.add_argument("--stochastic_round", action='store_true')
    parser.add_argument("--simulation", action='store_true')
    parser.add_argument("--cos_threshold", type=float, default=1)
    parser.add_argument("--gamma_proj", type=int, default=2)
    parser.add_argument("--queue_size", type=int, default=5)
    parser.add_argument("--act_quant", action='store_true')
    parser.add_argument('--act_bits', type=int, default=8)
    parser.add_argument('--act_group_size', type=int, default=64)
    parser.add_argument('--act_topk', type=int, default=2)
    parser.add_argument('--act_stochastic', action='store_true')
    parser.add_argument('--fp4', action = 'store_true')
    
    #stablespam
    parser.add_argument("--gamma1", type=float, default=0.85) # beta1 for Adafactor, GaLore_adafactor, (Q-)GaLore-adam or SGD
    parser.add_argument("--gamma2", type=float, default=0.999) # beta2 for (Q-)GaLore-adam
    parser.add_argument("--gamma3", type=float, default=0.999) # beta2 for (Q-)GaLore-adam
    parser.add_argument("--gamma4", type=float, default=0)
    parser.add_argument("--total_T", type=int, default=20000) # beta2 for (Q-)GaLore-adam
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--density", type=float, default=1.0)

    #tensorboard
    parser.add_argument("--set_tensorboard", action="store_true")
    parser.add_argument("--tensorboard_dir", type=str, default=None)

    #layernorm scaling
    parser.add_argument("--layernorm_scaling", default=False,action="store_true")
    parser.add_argument("--smoothswiglu", default=False,action="store_true")

    # Quantization
    parser.add_argument("--quest", default=False,action="store_true")
    parser.add_argument("--fp4atw", default=False,action="store_true") ## FP4 All the Way
    parser.add_argument("--quartet", default=False,action="store_true") ## Quartet

    parser.add_argument(
        "--w-quant", type=str, default="NoQuantizer", choices=QUANTIZER_CLASSES.keys()
    )
    
    parser.add_argument(
        "--w-quant-kwargs",
        type=json.loads,
        default="{}",
    )
    parser.add_argument(
        "--a-quant", type=str, default="NoQuantizer", choices=QUANTIZER_CLASSES.keys()
    )
    parser.add_argument(
        "--a-quant-kwargs",
        type=json.loads,
        default="{}",
    )
    parser.add_argument(
        "--g-quant", type=str, default="NoQuantizer", choices=QUANTIZER_CLASSES.keys()
    )
    parser.add_argument(
        "--g-quant-kwargs",
        type=json.loads,
        default="{}",
    )
    parser.add_argument(
        "--backward-scheme",
        type=str,
        default="EW_EtX",
        choices=["EW_EtX", "Q(E)W_Q(Et)X", "Q(E)Q(Wt)t_Q(Et)Q(Xt)t"],
    )
    parser.add_argument("--muonstep",    type=int, default=5)

    args = parser.parse_args(args)
    args = check_args_torchrun_main(args)
    return args

def max_train_tokens_to_number(max_train_tokens):
    if max_train_tokens.endswith("M"):
        return int(max_train_tokens.rstrip("M")) * 1_000_000
    elif max_train_tokens.endswith("B"):
        return int(max_train_tokens.rstrip("B")) * 1_000_000_000
    else:
        return int(max_train_tokens)

def check_args_torchrun_main(args):

    if args.save_dir is None:
        # use checkpoints / model name, date and time as save directory
        args.save_dir = f"checkpoints/{args.model_config.split('/')[-1].rstrip('.json')}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    os.makedirs(args.save_dir, exist_ok=True)

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Training for {args.num_training_steps} update steps")

    if args.continue_from is not None:
        assert os.path.exists(args.continue_from), f"--continue_from={args.continue_from} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported in torchrun_main.py. Use deepspeed_main.py instead (but it seems to have bugs)")

    return args
