# Setup

```sh
# assumes on Compute Canada Niagara
module load python/3.8.2

cd $SCRATCH
git clone git@github.com:stfxecutables/ABIDE-eig.git
cd ABIDE-eig

python virtualenv --no-download .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install antspyx matplotlib nibabel numpy pandas scikit-learn statsmodels tqdm pytest
```

# Data Download

After running setup:

```sh
source $SCRATCH/ABIDE-eig/.venv/bin/activate
cd $SCRATCH/ABIDE-eig/data/niis
python download_all.py
```

After downloading the data, you must also run

```sh
python $SCRATCH/ABIDE-eig/data/niis/log_shapes.py
```

to generate the `$SCRATCH/ABIDE-eig/data/niis/shapes.json` summary file needed for some later steps.


# To Run All Analyses

## Pre-Computations

These steps make various modified versions of the original files available.
This assumes you have run `cd $SCRATCH/job_scripts` first.

1. Precompute eigenimages (must be completed first)
   - `sbatch submit_precompute_eigimgs.sh`
2. Precompute other summary files (can be submitted in parallel)
   - `sbatch submit_precompute_roi_reductions.sh`
   - `sbatch submit_precompute_sequence_reductions.sh`

