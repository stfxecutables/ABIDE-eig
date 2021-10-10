#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=feat_extract%A__%a.out
#SBATCH --job-name=f_extract
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80


PROJECT=$SCRATCH/ABIDE-eig
VENV=$PROJECT/.venv/bin/activate
SCRIPT=$PROJECT/src/analysis/preprocess/extract_roi_features.py
export MPLCONFIGDIR=$SCRATCH/.mplconfig  # silence annoying matplotlib stuff

cd $PROJECT
source $VENV

echo "Starting to Extract ROI-based and full-brain features at $(date)"
python $SCRIPT && \
echo "Job done at $(date)"
