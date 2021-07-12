#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=eigimg%A__%a.out
#SBATCH --job-name=eigimg
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80


PROJECT=$SCRATCH/ABIDE-eig
VENV=$PROJECT/.venv/bin/activate
SCRIPT=$PROJECT/src/run/precompute_roi_reductions.py
export MPLCONFIGDIR=$SCRATCH/.mplconfig  # silence annoying matplotlib stuff

cd $PROJECT
source $VENV

echo "Starting to compute ROI reductions at $(date)"
python $SCRIPT && \
echo "Job done at $(date)"
