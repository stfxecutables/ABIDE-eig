#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=2:00:00
#SBATCH --signal=INT@300
#SBATCH --job-name=deep_sanity
#SBATCH --output=deep_sanity__%j.out
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=all_gpus
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

PROJECT=$HOME/projects/def-jlevman/dberger/ABIDE-eig
LOGS=$PROJECT/resnet_sanity_test

# echo "Setting up python venv"
# cd $SLURM_TMPDIR
# tar -xf $PROJECT/venv.tar .venv
source $PROJECT/.venv/bin/activate
PYTHON=$(which python)

echo "Job starting at $(date)"
tensorboard --logdir=$LOGS --host 0.0.0.0 &
$PYTHON $PROJECT/src/analysis/predict/models.py && \
echo "Job done at $(date)"
