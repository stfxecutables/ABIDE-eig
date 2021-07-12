#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time=24:00:00
#SBATCH --job-name=eigimg
#SBATCH --output=eigimg%A__%a.out
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

PROJECT=$SCRATCH/ABIDE-eig
SCRIPT=$PROJECT/src/eigenimage/compute_batch.py
export MPLCONFIGDIR=$SCRATCH/.mplconfig  # silence annoying matplotlib stuff

module load python/3.8.2
cd $SLURM_TMPDIR
tar -xf $PROJECT/venv.tar .venv
source .venv/bin/activate
PYTHON=$(which python)

echo "Job starting at $(date)"
$PYTHON $SCRIPT --batch=$SLURM_ARRAY_TASK_ID && \
echo "Job done at $(date)"
