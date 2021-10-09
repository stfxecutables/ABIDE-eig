import os
import subprocess
import sys
import traceback
from argparse import ArgumentParser
from io import StringIO
from pathlib import Path
from time import strftime
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PIPELINE = "cpac"
STRATEGY = "filt_noglobal"  # see http://preprocessed-connectomes-project.org/abide/Pipelines.html
NII_EXT = "nii.gz"
ROI_EXT = "1D"
FUNC_DERIVATIVES = "func_preproc"
ROI_ATLASES = ["cc200", "cc400", "ho"]
ROI_DERIVATIVES = [f"rois_{atlas}" for atlas in ROI_ATLASES]
FUNC_TEMPLATE = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/"
    f"{PIPELINE}/{STRATEGY}/{FUNC_DERIVATIVES}/{{fname}}_{FUNC_DERIVATIVES}.{NII_EXT}"
)
ROI_TEMPLATE = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/"
    f"{PIPELINE}/{STRATEGY}/{{roi_derivative}}/{{fname}}_{{roi_derivative}}.{ROI_EXT}"
)

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
ROI_OUTS = [Path(__file__).resolve().parent / f"rois_cpac_{atlas}" for atlas in ROI_ATLASES]
NII_OUT = Path(__file__).resolve().parent / "nii_cpac"
if not NII_OUT.exists():
    os.makedirs(NII_OUT, exist_ok=True)
for out in ROI_OUTS:
    os.makedirs(out, exist_ok=True)
SHAPES_DATA = NII_OUT / "shapes.json"

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

HEINSFELD = pd.read_csv(
    StringIO(
        """site,    asd_age_mean, asd_age_sd,   asd_n_M, asd_n_F, td_age_mean,  td_age_sd,  td_n_M, td_n_F, FD_quality
CALTECH,         27.4,       10.3,        15,       4,        28.0,       10.9,      14,      4,  0.07
CMU,             26.4,        5.8,        11,       3,        26.8,        5.7,      10,      3,  0.29
KKI,             10.0,        1.4,        16,       4,        10.0,        1.2,      20,      8,  0.17
LEUVEN,          17.8,        5.0,        26,       3,        18.2,        5.1,      29,      5,  0.09
MAX_MUN,         26.1,       14.9,        21,       3,        24.6,        8.8,      27,      1,  0.13
NYU,             14.7,        7.1,        65,      10,        15.7,        6.2,      74,     26,  0.07
OHSU,            11.4,        2.2,        12,       0,        10.1,        1.1,      14,      0,  0.10
OLIN,            16.5,        3.4,        16,       3,        16.7,        3.6,      13,      2,  0.18
PITT,            19.0,        7.3,        25,       4,        18.9,        6.6,      23,      4,  0.15
SBL,             35.0,       10.4,        15,       0,        33.7,        6.6,      15,      0,  0.16
SDSU,            14.7,        1.8,        13,       1,        14.2,        1.9,      16,      6,  0.09
STANFORD,        10.0,        1.6,        15,       4,        10.0,        1.6,      16,      4,  0.11
TRINITY,         16.8,        3.2,        22,       0,        17.1,        3.8,      25,      0,  0.11
UCLA,            13.0,        2.5,        48,       6,        13.0,        1.9,      38,      6,  0.19
UM,              13.2,        2.4,        57,       9,        14.8,        3.6,      56,     18,  0.14
USM,             23.5,        8.3,        46,       0,        21.3,        8.4,      25,      0,  0.14
YALE,            12.7,        3.0,        20,       8,        12.7,        2.8,      20,      8,  0.11
""".replace(
            " ", ""
        )
    ),
    dtype=DTYPES,
)
HEINSFELD.index = HEINSFELD.site
HEINSFELD.drop(columns=["site", "FD_quality"], inplace=True)


def combine_sites(s: str) -> str:
    return s.replace("_1", "").replace("_2", "")


def get_site_counts(csv: DataFrame) -> DataFrame:
    readable = csv.copy()
    readable.SEX = readable.SEX.apply(lambda x: "M" if str(x) == "1" else "F")
    readable.DX_GROUP = readable.DX_GROUP.apply(lambda x: "ASD" if str(x) == "1" else "TD")
    readable.DX_GROUP = readable.DX_GROUP.astype("category")
    readable.SEX = readable.SEX.astype("category")
    readable.SITE_ID = readable.SITE_ID.apply(combine_sites)
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


def compare_counts(counts: DataFrame, full: bool = False) -> DataFrame:
    discrep = counts.round(1) - HEINSFELD
    if not full:
        discrep = discrep.filter(regex="_n_")
    return discrep


def filter_to_1035(csv: DataFrame) -> DataFrame:
    all_subj = csv.copy()
    undiagnosed = csv.DSM_IV_TR.isna()
    QC_COLS = [
        "qc_rater_1",
        "qc_anat_rater_2",
        "qc_func_rater_2",
        "qc_anat_rater_3",
        "qc_func_rater_3",
    ]
    qc = csv.loc[:, QC_COLS]
    keep = ~(qc == "fail").any(axis=1)
    undiag = csv.loc[~undiagnosed]
    nameless = csv.FILE_ID == "no_filename"

    our_counts = get_site_counts(all_subj)
    undiag_counts = get_site_counts(undiag)
    nameless_counts = get_site_counts(csv.loc[~nameless])

    pd.options.display.max_rows = 900
    print("Full data discrepancies from Heinsfeld")
    print(compare_counts(our_counts))
    print("Undiagnosed data discrepancies from Heinsfeld")
    print(compare_counts(undiag_counts))
    print("Nameless data discrepancies from Heinsfeld")
    print(compare_counts(nameless_counts, full=True))

    # NOTE: from above it is clear that what all the 1035 subject papers *actually* do is
    # remove subjects that have "no_filename" as the name.
    subj_1035 = csv.loc[~nameless].copy()
    return subj_1035


def download_csv() -> DataFrame:
    CSV_FILE = Path(__file__).resolve().parent / "Phenotypic_V1_0b_preprocessed1.csv"
    CSV_WEB_PATH = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
    if not CSV_FILE.exists():
        urlretrieve(CSV_WEB_PATH, CSV_FILE)
    csv = pd.read_csv(CSV_FILE, header=0, na_values="-9999")
    csv = filter_to_1035(csv)
    csv = csv.rename(columns={"SUB_ID": "sid", "FILE_ID": "fname"})
    csv.index = csv.sid
    csv.drop(columns="sid", inplace=True)
    return csv


def download_file(url: str, outdir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        f"cd {outdir} && wget --timestamp --continue {url}",
        capture_output=False,
        check=True,
        shell=True,
    )


def download_rois() -> None:
    start = strftime("%Y-%h-%d--%H:%M")
    LOGFILE = Path(__file__).resolve().parent / f"rois_{start}.log"
    if not LOGFILE.exists():
        LOGFILE.touch()
    csv = download_csv()
    fids = csv.fname.to_list()
    for roi_deriv, roi_out in zip(ROI_DERIVATIVES, ROI_OUTS):
        for fid in tqdm(fids, desc=f"Downloading {roi_deriv}"):
            url = ROI_TEMPLATE.format(roi_derivative=roi_deriv, fname=fid)
            try:
                download_file(url=url, outdir=roi_out)
            except subprocess.CalledProcessError as e:
                with open(LOGFILE, "w+") as stream:
                    err = traceback.format_exc()
                    print(f"Error downloading {url}:", file=stream)
                    print("Traceback: ", file=stream)
                    print(err, file=stream)
                    print("stderr:", file=stream)
                    print(e.stderr, stream)
                    print("stdout:", file=stream)
                    print(e.stdout, stream)


def download_fmri() -> None:
    start = strftime("%Y-%h-%d--%H:%M")
    LOGFILE = Path(__file__).resolve().parent / f"rois_{start}.log"
    if not LOGFILE.exists():
        LOGFILE.touch()
    csv = download_csv()
    fids = csv.fname.to_list()
    for fid in tqdm(fids, desc="Downloading fMRI images"):
        url = FUNC_TEMPLATE.format(fname=fid)
        try:
            download_file(url=url, outdir=NII_OUT)
        except subprocess.CalledProcessError as e:
            with open(LOGFILE, "w+") as stream:
                err = traceback.format_exc()
                print(f"Error downloading {url}:", file=stream)
                print("Traceback: ", file=stream)
                print(err, file=stream)
                print("stderr:", file=stream)
                print(e.stderr, stream)
                print("stdout:", file=stream)
                print(e.stdout, stream)


def download_fmri_subset() -> None:
    csv = download_csv().iloc[:, 3:]
    csv = csv.sort_values(by="subject")
    # note `stratify` argument needs a 1D vector, so we just hack here
    # and concatenate the stratify labels. We don't need to look at shapes
    # or scan-info, since they are the same within a site
    stratify = csv.SITE_ID.astype(str) + csv.DX_GROUP.astype(str)
    info = pd.DataFrame(
        {
            "sid": csv.index.to_series(),
            "fname": csv.fname.values,
            "site": csv.SITE_ID.values,
            "cls": csv.DX_GROUP.apply(lambda s: "ASD" if str(s) == "1" else "TD"),
        }
    )
    train, _ = train_test_split(info, train_size=200, stratify=stratify, random_state=333)
    train = train.sort_values(by=["site", "cls"])
    print(train.drop(columns="fname").groupby(["site", "cls"]).count())
    fids = train.fname.to_list()
    # NOTE 200 subjects is roughly 21.5 GB (roughly 107MB per subject)
    res = input("Given subject distribution above, proceed to download? [y/N]\n")
    if res.lower() != "y":
        sys.exit()
    for fid in tqdm(fids):
        url = FUNC_TEMPLATE.format(fname=fid)
        subprocess.run(
            f"cd {NII_OUT} && wget --timestamping --continue {url}",
            capture_output=False,
            check=False,
            shell=True,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--subsample", action="store_true")
    parser.add_argument("--rois", action="store_true")
    parser.add_argument("--fmri", action="store_true")
    args = parser.parse_args()
    subsample = args.subsample
    if args.rois:
        download_rois()
    if args.subsample:
        download_fmri_subset()
    elif args.fmri:
        download_fmri()
    else:
        print("Done.")
