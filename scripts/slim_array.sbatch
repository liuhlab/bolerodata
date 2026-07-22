#!/bin/bash
# Slim all Bolero checkpoints in parallel (drop optimizer state, keep parameters).
# Submit AFTER `python scripts/slim_checkpoint.py plan` has written dist/slim_tasks.tsv:
#   sbatch --array=1-$(( $(wc -l < dist/slim_tasks.tsv) - 1 ))%20 scripts/slim_array.sbatch
#SBATCH --job-name=bolero-slim
#SBATCH --partition=quick_preemptible
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/zhoulab/hanliu/huggingfase_model/_logs/slim_%A_%a.out
set -uo pipefail

# quick_preempt QOS allows only 4 running / 8 submitted jobs, so submit a SMALL
# array (e.g. 1-4) and let each task stride-partition the whole task list:
#   sbatch --array=1-4 scripts/slim_array.sbatch
BPY=/large_storage/zhoulab/hanliu/pkg/liuhlab/bolero/.pixi/envs/default/bin/python
SCRIPT=/large_storage/zhoulab/hanliu/pkg/liuhlab/bolerodata/scripts/slim_checkpoint.py

echo "task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname) @ $(date)"
"$BPY" "$SCRIPT" run --index "${SLURM_ARRAY_TASK_ID}" --stride "${SLURM_ARRAY_TASK_COUNT}"
