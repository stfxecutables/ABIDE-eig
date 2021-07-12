#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=roi_pred%A__%a.out
#SBATCH --job-name=roi_pred
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80


PROJECT=$SCRATCH/ABIDE-eig
VENV=$PROJECT/.venv/bin/activate
SCRIPT=$PROJECT/src/run/compute_roi_predictions.py
export MPLCONFIGDIR=$SCRATCH/.mplconfig  # silence annoying matplotlib stuff

cd $PROJECT
source $VENV

echo "Starting to compute ROI predictions at $(date)"
python $SCRIPT --silent && \
echo "Job done at $(date)"
