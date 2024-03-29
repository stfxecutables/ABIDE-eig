import traceback
from os import system
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from tqdm import tqdm

NII_DIR = Path(__file__).resolve().parent
FNAME = "{}_func_minimal.nii.gz"
LINK_TEMPLATE = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/func_minimal/{}"
)
CSV_FILE = Path(__file__).resolve().parent.parent / "Phenotypic_V1_0b_preprocessed1.csv"
CSV_WEB_PATH = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
INFO_COLUMNS = [  # note these are almost all NaN or Nones
    "FILE_ID",
    "SUB_ID",
    "DX_GROUP",  # 1 = Autism, 2 = Control
    "DSM_IV_TR",  # 0 = Control, 1 = Autism, 2 = Aspergers, 3 = PDD-NOS, 4 = Apergers or PDD-NOS
    "AGE_AT_SCAN",
    "SEX",  # 1 = Male, 2 = Female
    "HANDEDNESS_SCORES",
    "FIQ",  # Full IQ
    "VIQ",  # Verval IQ
    "PIQ",  # Performance IQ
    "SCQ_TOTAL",  # Social Communication Questionnare total score
    "AQ_TOTAL",  # Total raw score of Autism Quotient
    "COMORBIDITY",  # str: other comorbidities
    "CURRENT_MED_STATUS",  # 0 = None, 1 = taking medication
    "OFF_STIMULANTS_AT_SCAN",  # 0 = No, 1 = Yes
    "WISC_IV_MATRIX_SCALED",  # WISC-IV Matrix Reasoning score (scaled)
]
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


def download_csv() -> None:
    if not CSV_FILE.exists():
        urlretrieve(CSV_WEB_PATH, CSV_FILE)


if __name__ == "__main__":
    download_csv()

    csv = pd.read_csv(CSV_FILE, header=0, na_values="-9999").loc[:, INFO_COLUMNS]
    detailed = csv.loc[:, USEFUL_COLUMNS].dropna()
    known_dx = detailed["DSM_IV_TR"] <= 2
    is_adult = detailed["AGE_AT_SCAN"] >= 18.0
    is_sortof_smart = detailed["FIQ"] >= 90
    # filters = known_dx & is_adult & is_sortof_smart
    # filters = known_dx

    # detailed = detailed.loc[filters, :].copy()

    males = detailed.loc[detailed.SEX == 1.0, :]
    females = detailed.loc[detailed.SEX == 2.0, :]

    # limited_locations = np.array(
    #     [("NYU" in s) or ("CMU" in s) for s in males.FILE_ID.to_list()], dtype=bool
    # )
    # males = males.loc[limited_locations, :].copy()

    female_IQ_max = females.loc[:, ["FIQ", "VIQ", "PIQ"]].max(axis=0).max()
    is_matched_male = (
        (males.FIQ <= female_IQ_max) & (males.VIQ <= female_IQ_max) & (males.PIQ <= female_IQ_max)
    )

    males = males.loc[is_matched_male, :].copy()

    is_aspy_male = males.DSM_IV_TR == 2.0
    asp_males = males.loc[
        is_aspy_male, :
    ].copy()  # only 4 aspy females, so make sure we have some aspy males
    autistic_males = males.loc[(males.DX_GROUP == 1) & (~is_aspy_male), :].copy()
    autistic_females = females.loc[females.DX_GROUP == 1, :].copy()

    control_males = males.loc[males.DX_GROUP == 2, :].copy()
    control_males = control_males.iloc[: len(autistic_males), :].copy()
    control_females = females.loc[females.DX_GROUP == 2, :].copy()

    all_subjs = pd.concat(
        [asp_males, autistic_males, control_males, autistic_females, control_females], axis=0
    )
    ids_and_labels = all_subjs.loc[:, ["FILE_ID", "SUB_ID", "DX_GROUP"]]
    ids_and_labels.to_json(Path(".") / "subj_data.json")

    total = len(all_subjs)
    print(f"Total number of subjects - {total}:")
    print(f"Asperger males:   {len(asp_males)}")
    print(f"Autistic males:   {len(autistic_males)}")
    print(f"Control males:    {len(control_males)}")
    print(f"Autistic females: {len(autistic_females)}")
    print(f"Control females:  {len(control_females)}")

    totalsize = str(310 * total / 1000) + "GB"  # est. 310 MB per file
    print(f"Estimated size of files: {totalsize}.")

    # control_males_reduced = control_males.iloc[: len(autistic_males), :].copy()
    # response = input("Continue to download? [y/N]: ")
    # if str(response).upper() != "Y":
    #     sys.exit()

    print(all_subjs.describe())
    ids, sids = all_subjs["FILE_ID"], all_subjs["SUB_ID"]

    downloaded = []
    for i, (id, sid) in tqdm(enumerate(zip(ids, sids)), total=len(ids)):
        try:
            fname = FNAME.format(id)
            outfile = NII_DIR / fname
            if outfile.exists():
                downloaded.append(outfile)
                continue
            link = LINK_TEMPLATE.format(fname)
            system(f"cd {NII_DIR} && wget {link}")
            print(f"Downloaded file {i}/{len(ids)}. {100*(i+1)/len(ids):2.1f}% done.")
            downloaded.append(outfile)
        except BaseException as e:
            print(f"Problem downloading file with FILE_ID: {id}")
            traceback.print_exc()
            print(e)
            print(f"Moving to next file. {100*(i+1)/len(ids):2.1f}% done.")
#        if len(downloaded) >= 5:
#            break
