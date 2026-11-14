#!/usr/bin/env bash
#SBATCH --job-name=refer-paper4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/user1/NKU/Refer-Agent-reproduction/runs/slurm-%j.log
set -euo pipefail
cd /home/user1/NKU/Refer-Agent-reproduction
export PATH=/home/user1/miniconda3/envs/sec/bin:$PATH
export PYTHONPATH="$PWD:$PWD/sam2"
export CUDA_HOME=/usr/local/cuda-12.4
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TZ=Asia/Shanghai
run_arguments=("$@")
runner_pid=''
continue_after_time_limit() {
    trap '' USR1
    if [[ -n "$runner_pid" ]] && kill -0 "$runner_pid" 2>/dev/null; then
        kill -TERM -- "-$runner_pid" 2>/dev/null || true
        wait "$runner_pid" 2>/dev/null || true
        sbatch --parsable "$PWD/tools/reasonvos.slurm" "${run_arguments[@]}"
        exit 0
    fi
}
trap continue_after_time_limit USR1
setsid python -u tools/run_reasonvos.py "${run_arguments[@]}" &
runner_pid=$!
set +e
wait "$runner_pid"
exit_code=$?
set -e
trap - USR1
exit "$exit_code"
