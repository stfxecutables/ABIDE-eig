#!/bin/bash
source .venv/bin/activate
python src/analysis/predict/deep_learning/generate_jobscripts.py \
    --gpu=1 \
    --max_epochs=125 \
    --batch_size=25 \
    --val_batch_size=20 \
    --slice_start=125 \
    --slice_end=175 \
    --is_eigimg
sbatch job_scripts/submit_htune_eigimg.sh
