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

1. Compute eigenimages
   - `sbatch submit_eigimg.sh`
2. Compute ROI summary files
   - `sbatch submit_compute_roi_summaries.sh`

