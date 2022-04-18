import os
import subprocess
import sys
import traceback
from argparse import ArgumentParser
from io import StringIO
from pathlib import Path
from shutil import copyfile
from time import strftime
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

NII_OUT = Path(__file__).resolve().parent.parent.parent.parent / "data/nii_cpac"
SHAPES_DATA = NII_OUT / "shapes.json"

USEFUL_COLUMNS = [  # note these are almost all NaN or Nones
    "FILE_ID",
    "SUB_ID",
    "DX_GROUP",  # 1 = Autism, 2 = Control
    "DSM_IV_TR",  # 0 = Control, 1 = Autism, 2 = Aspergers, 3 = PDD-NOS, 4 = Apergers or PDD-NOS
    "AGE_AT_SCAN",
    "SEX",
    "FIQ",  # Full IQ
    "VIQ",  # Verval IQ
    "PIQ",  # Performance IQ
    "CURRENT_MED_STATUS",  # 0 = None, 1 = taking medication
]
QC_COLS = [
    "qc_rater_1",
    "qc_anat_rater_2",
    "qc_func_rater_2",
    "qc_anat_rater_3",
    "qc_func_rater_3",
]

# we compare to Table 1 of Heinsfeld, A. S., Franco, A. R., Craddock, R. C., Buchweitz, A., &
# Meneguzzi, F. (2018). Identification of autism spectrum disorder using deep learning and the ABIDE
# dataset. NeuroImage: Clinical, 17, 16–23. https://doi.org/10.1016/j.nicl.2017.08.017 In so doing
# we see excluded are:
#
# CALTECH: 1 M TD CMU: None
DTYPES = {
    "site": str,
    "asd_age_mean": float,
    "asd_age_sd": float,
    "asd_n_M": int,
    "asd_n_F": int,
    "td_age_mean": float,
    "td_age_sd": float,
    "td_n_M": int,
    "td_n_F": int,
}
DTYPES2 = {
    # "site": "str",
    "asd_age_mean": "float",
    "asd_age_sd": "float",
    "asd_n_M": "int",
    "asd_n_F": "int",
    "td_age_mean": "float",
    "td_age_sd": "float",
    "td_n_M": "int",
    "td_n_F": "int",
}


def filter_to_1035(csv: DataFrame) -> DataFrame:
    nameless = csv.FILE_ID == "no_filename"
    return csv.loc[~nameless].copy()


def download_csv() -> DataFrame:
    CSV_FILE = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "Phenotypic_V1_0b_preprocessed1.csv"
    )
    CSV_WEB_PATH = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
    if not CSV_FILE.exists():
        urlretrieve(CSV_WEB_PATH, CSV_FILE)
    csv = pd.read_csv(CSV_FILE, header=0, na_values="-9999")
    csv = filter_to_1035(csv)
    csv = csv.rename(columns={"SUB_ID": "sid", "FILE_ID": "fname"})
    csv.index = csv.sid
    csv.drop(columns="sid", inplace=True)
    csv = csv.iloc[:, 3:]
    return csv


def combine_sites(s: str) -> str:
    return s.replace("_1", "").replace("_2", "")


def get_site_info(csv: DataFrame) -> DataFrame:
    KEEP = [
        "SITE_ID",
        "MALE (%)",
        "fname",
        "DX_GROUP",
        "DSM_IV_TR",
        "AGE_AT_SCAN",
        "FIQ",
        "VIQ",
        "PIQ",
        "func_quality",
    ]
    SORT = [
        "MALE (%)",
        "DSM_IV_TR",
        "AGE_AT_SCAN",
        "T",
        "TR",
        "FIQ",
        "VIQ",
        "PIQ",
        "func_quality",
    ]
    readable = csv.copy()
    readable.SEX = readable.SEX.apply(lambda x: 100 if str(x) == "1" else 0)
    readable.rename(columns={"SEX": "MALE (%)"}, inplace=True)
    readable.DX_GROUP = readable.DX_GROUP.apply(lambda x: "ASD" if str(x) == "1" else "TD")
    readable.DX_GROUP = readable.DX_GROUP.astype("category")
    # readable.SEX = readable.SEX.astype("category")
    readable.SITE_ID = readable.SITE_ID.apply(combine_sites)
    readable = readable.loc[:, KEEP]

    shapes = pd.read_json(SHAPES_DATA)
    shapes.index = shapes.index.to_series().apply(
        lambda s: int(str(s).replace("_func_preproc.nii", "").split("_")[-1])
    )
    shapes = shapes.drop(columns="shape")
    shapes.rename(columns=dict(dt="TR"), inplace=True)

    info = pd.merge(readable, shapes, on=readable.index)
    means = info.groupby(["SITE_ID", "DX_GROUP"]).mean().drop(columns="key_0").loc[:, SORT]
    counts = info.groupby(["SITE_ID", "DX_GROUP"]).count().loc[:, "T"].rename({"T": "N"})
    return means, counts

    SITES = sorted(readable.SITE_ID.unique())
    our_demos = pd.DataFrame(
        index=pd.Series(SITES, name="site"),
        columns=[
            "asd_age_mean",
            "asd_age_sd",
            "asd_n_M",
            "asd_n_F",
            "td_age_mean",
            "td_age_sd",
            "td_n_M",
            "td_n_F",
        ],
    )
    for site in sorted(readable.SITE_ID.unique()):
        subjects = readable.loc[readable.SITE_ID == site]
        counts = subjects.groupby(["DX_GROUP", "SEX"]).count().iloc[:, 0]
        age_means = subjects.groupby(["DX_GROUP"])["AGE_AT_SCAN"].mean()
        age_sds = subjects.groupby(["DX_GROUP"])["AGE_AT_SCAN"].std(ddof=1)

        our_demos.loc[site, "asd_age_mean"] = age_means.loc["ASD"]
        our_demos.loc[site, "asd_age_sd"] = age_sds.loc["ASD"]
        our_demos.loc[site, "asd_n_M"] = counts.loc["ASD", "M"]
        our_demos.loc[site, "asd_n_F"] = counts.loc["ASD", "F"]
        our_demos.loc[site, "td_age_mean"] = age_means.loc["TD"]
        our_demos.loc[site, "td_age_sd"] = age_sds.loc["TD"]
        our_demos.loc[site, "td_n_M"] = counts.loc["TD", "M"]
        our_demos.loc[site, "td_n_F"] = counts.loc["TD", "F"]
    for col in our_demos.columns:
        our_demos[col] = our_demos[col].astype(DTYPES2[col])
    return our_demos


if __name__ == "__main__":
    subj = download_csv()
    means, counts = get_site_info(subj)
    print(counts)
    print(means.round(1))
