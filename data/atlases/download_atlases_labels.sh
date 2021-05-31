#!/bin/bash
# wget \
#    --no-check-certificate \
#    --recursive \
#    --no-parent \
#    --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36" \
#    --accept "*atlas*.nii.gz" \
#    -P atlases \
#    'http://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/'
#
# wget --recursive --no-parent --accept "*labels.csv" -P labels https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/aal_roi_atlas.nii.gz
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/aal_labels.csv
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/ez_roi_atlas.nii.gz
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/ez_labels.csv
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/ho_roi_atlas.nii.gz
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/ho_labels.csv
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/tt_roi_atlas.nii.gz
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/tt_labels.csv
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/dos160_roi_atlas.nii.gz
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/dos160_labels.csv
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/cc200_roi_atlas.nii.gz
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/CC200_ROI_labels.csv
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/cc400_roi_atlas.nii.gz
wget https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/CC400_ROI_labels.csv
