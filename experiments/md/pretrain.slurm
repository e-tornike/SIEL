#!/bin/bash
# Adapted from https://github.com/stas00/ml-engineering/blob/master/orchestration/slurm/launchers/accelerate-launcher.slurm

# this is a 1 node SLURM script using `accelerate` launcher

#SBATCH --job-name=XLM-R-L-700k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per node
#SBATCH --gres=gpu:A100:4                 # EDIT this if it's not 4-gpus per node
#SBATCH --time=48:00:00               # EDIT the desired runtime
#SBATCH --exclusive
#SBATCH --partition=single      # EDIT to the desired partition name
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL

echo "START TIME: $(date)"

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x

# EDIT the conda evn and any startup scripts
# cd into the project directory & activate the virtual environment
# eval $(pdm venv activate 3.9)

LOG_PATH="main_log.txt"

# EDIT the path to accelerate config file
ACCELERATE_CONFIG_FILE=~/.cache/huggingface/accelerate/default_config.yaml

# EDIT if it's not 4-gpus per node
GPUS_PER_NODE=4
NNODES=$SLURM_NNODES

# define the node 0 hostname:port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

LAUNCHER="pdm run python -u -m accelerate.commands.launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): --tee 3 \
    "

PROGRAM="\
src/run_mlm_no_trainer.py \
--model_name_or_path FacebookAI/xlm-roberta-large \
--train_file ./data/s44k/texts_train.txt \
--validation_file ./data/s44k/texts_val.txt \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--max_train_steps 700000 \
--line_by_line True \
--max_seq_length 512 \
--output_dir ./models/ssoar_plm/XLMR_large_steps=700k_batch=8_gpus=4
"
export CMD="$LAUNCHER $PROGRAM"

echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"