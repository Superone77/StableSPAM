

# module purge
# module load 2023
# source activate spam
# Your job starts in the directory where you call sbatch
WORK_DIR=/local/mnt2/workspace/wanqi/test/optimizer_8bit/StableSPAM
NAME=9-stablespam-fp4atw
TENSORBOARD_DIR=$WORK_DIR/tensorboard_dir/$NAME
LOG_DIR=$WORK_DIR/logs/$NAME.log
export W_QUANT="HalfHadamardTrustQuantizer"
export A_QUANT="HalfHadamardTrustQuantizer"
export W_QUANT_KWARGS="{\"bits\": 4}"
export A_QUANT_KWARGS="{\"bits\": 4}"
cd $WORK_DIR
rm -rf $TENSORBOARD_DIR
mkdir $TENSORBOARD_DIR
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli login --token $HF_TOKEN
echo $HF_HOME

huggingface-cli whoami
export DELAYED_SCALED_SWIGLU="false"
export DETACH_SCALED_SWIGLU="true"
export SCALED_SWIGLU="false"
export BLOCK_SIZE="16"
export SCALE_FORMAT="e4m3"
# export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=gpu main_pretrain.py \
    --model_config configs/llama_9m.json \
    --eval_every 1000 \
    --save_every 5000 \
    --dtype bfloat16 \
    --batch_size 256 \
    --total_batch_size 512 \
    --lr 0.0004 \
    --warmup_steps 2000 \
    --num_training_steps 5000 \
    --optimizer stablespam \
    --weight_decay 0 \
    --name stablespam_350_fp4_500_0.9_0.7_4e-4 \
    --save_dir $WORK_DIR/tmp/checkpoint/$NAME/ \
    --unset_wandb \
    --set_tensorboard \
    --fp4atw \
    --gamma1 0.7 \
    --gamma2 0.99999 \
    --gamma3 0.999 \
    --update_proj_gap 1000 \
    --tensorboard_dir $TENSORBOARD_DIR  > $LOG_DIR 2>&1 

# tensorboard --logdir=runs
# --continue_from 从已有的checkpoint继续训练
#