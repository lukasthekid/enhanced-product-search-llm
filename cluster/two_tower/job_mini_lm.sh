#!/bin/bash
#SBATCH --job-name=mini_lm_job         # Job name
#SBATCH --output=output_mini_lm_%j.log           # Output file
#SBATCH --error=error_mini_lm_%j.log             # Error file
#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1                        # select number of nodes
#SBATCH --ntasks-per-node=32             # select number of tasks per node
#SBATCH --mem=120GB                       # memory size required per node
#SBATCH --time=168:00:00                  # Time limit hrs:min:sec

echo "Starting job script at $(date)"
echo "Running on node(s): $(hostname)"
echo "Current directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

start_time=$(date +%s)
# Run your Python script
pip install --upgrade pip
#pip install -U tensorflow[and-cuda] tensorflow_recommenders tf_keras
pip install -U tensorflow[and-cuda] tensorflow_recommenders sentence_transformers tf_keras
#pip install numpy==1.26.0
#nvidia-smi
#nvcc --version

python3 mini-lm-eval.py
end_time=$(date +%s)
elapsed_time=$(($end_time - $start_time))

echo "Job completed at $(date)"
echo "Elapsed time: $(($elapsed_time / 3600))h $((($elapsed_time / 60) % 60))m $(($elapsed_time % 60))s"