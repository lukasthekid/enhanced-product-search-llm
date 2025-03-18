#!/bin/bash
#SBATCH --job-name=parameter_tuning_job         # Job name
#SBATCH --output=output_tuning_%j.log           # Output file
#SBATCH --error=error_tuning_%j.log             # Error file
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
#pip install --upgrade numpy pytz typing-extensions requests transformers python-dateutil -U
#pip3 install datasets accelerate -U
pip install evaluate rouge_score torch torchvision torchaudio transformers tf-keras

python hyper_parameter_tuning.py
end_time=$(date +%s)
elapsed_time=$(($end_time - $start_time))

echo "Job completed at $(date)"
echo "Elapsed time: $(($elapsed_time / 3600))h $((($elapsed_time / 60) % 60))m $(($elapsed_time % 60))s"