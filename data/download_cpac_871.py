import os
import subprocess
import sys
import traceback
from os import system
from pathlib import Path
from time import strftime
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

# Available, from, https://ars.els-cdn.com/content/image/1-s2.0-S1053811916305924-mmc1.pdf
# fmt: off
SIDS = [
    50003, 50152, 50290, 50404, 50531, 50728, 50988, 51083, 51194, 51315,
    50004, 50153, 50291, 50405, 50532, 50730, 50989, 51084, 51197, 51318,
    50005, 50156, 50292, 50406, 50551, 50731, 50990, 51085, 51198, 51319,
    50006, 50157, 50293, 50407, 50552, 50733, 50991, 51086, 51201, 51320,
    50007, 50158, 50294, 50408, 50555, 50735, 50992, 51087, 51202, 51321,
    50008, 50159, 50295, 50410, 50557, 50737, 50993, 51088, 51203, 51322,
    50010, 50160, 50297, 50411, 50558, 50738, 50994, 51089, 51204, 51323,
    50011, 50161, 50298, 50412, 50561, 50739, 50995, 51090, 51205, 51325,
    50012, 50162, 50300, 50413, 50563, 50740, 50996, 51091, 51206, 51326,
    50013, 50163, 50301, 50414, 50565, 50741, 50997, 51093, 51207, 51327,
    50014, 50164, 50302, 50415, 50568, 50742, 50999, 51094, 51208, 51328,
    50015, 50167, 50304, 50416, 50569, 50743, 51000, 51095, 51210, 51329,
    50016, 50168, 50308, 50417, 50570, 50744, 51001, 51096, 51211, 51330,
    50020, 50169, 50310, 50418, 50571, 50745, 51002, 51097, 51212, 51331,
    50022, 50170, 50312, 50419, 50572, 50748, 51003, 51098, 51214, 51332,
    50023, 50171, 50314, 50421, 50573, 50749, 51006, 51099, 51215, 51333,
    50024, 50182, 50315, 50422, 50574, 50750, 51007, 51100, 51216, 51334,
    50025, 50183, 50318, 50424, 50575, 50751, 51008, 51101, 51217, 51335,
    50026, 50184, 50319, 50425, 50576, 50752, 51009, 51102, 51218, 51336,
    50027, 50186, 50320, 50426, 50577, 50754, 51010, 51103, 51219, 51338,
    50028, 50187, 50321, 50427, 50578, 50755, 51011, 51104, 51220, 51339,
    50030, 50188, 50324, 50428, 50601, 50756, 51012, 51105, 51221, 51340,
    50031, 50189, 50325, 50433, 50602, 50757, 51013, 51106, 51222, 51341,
    50032, 50190, 50327, 50434, 50603, 50772, 51014, 51107, 51223, 51342,
    50033, 50193, 50329, 50435, 50604, 50773, 51015, 51109, 51224, 51343,
    50034, 50194, 50330, 50436, 50606, 50774, 51016, 51110, 51225, 51344,
    50035, 50195, 50331, 50437, 50607, 50775, 51017, 51111, 51226, 51345,
    50036, 50196, 50332, 50438, 50608, 50776, 51018, 51112, 51228, 51346,
    50037, 50198, 50333, 50439, 50612, 50777, 51019, 51113, 51229, 51347,
    50038, 50199, 50334, 50440, 50613, 50778, 51020, 51114, 51230, 51349,
    50039, 50200, 50335, 50441, 50614, 50780, 51021, 51116, 51231, 51350,
    50040, 50201, 50336, 50442, 50615, 50781, 51023, 51117, 51234, 51351,
    50041, 50202, 50337, 50443, 50616, 50782, 51024, 51118, 51235, 51354,
    50042, 50203, 50338, 50444, 50619, 50783, 51025, 51122, 51236, 51356,
    50043, 50204, 50339, 50445, 50620, 50786, 51026, 51123, 51237, 51357,
    50044, 50205, 50340, 50446, 50621, 50790, 51027, 51124, 51239, 51359,
    50045, 50206, 50341, 50447, 50622, 50791, 51028, 51126, 51240, 51360,
    50046, 50208, 50342, 50448, 50623, 50792, 51029, 51127, 51241, 51361,
    50047, 50210, 50343, 50449, 50624, 50796, 51030, 51128, 51248, 51362,
    50048, 50213, 50344, 50453, 50625, 50797, 51032, 51129, 51249, 51363,
    50049, 50214, 50345, 50463, 50626, 50798, 51033, 51130, 51250, 51364,
    50050, 50215, 50346, 50466, 50627, 50799, 51034, 51131, 51251, 51365,
    50051, 50217, 50347, 50467, 50628, 50800, 51035, 51132, 51252, 51369,
    50052, 50232, 50348, 50468, 50642, 50801, 51036, 51133, 51253, 51370,
    50053, 50233, 50349, 50469, 50644, 50803, 51038, 51134, 51254, 51373,
    50054, 50234, 50350, 50470, 50647, 50807, 51039, 51135, 51255, 51461,
    50056, 50236, 50351, 50477, 50648, 50812, 51040, 51136, 51256, 51463,
    50057, 50237, 50352, 50480, 50649, 50814, 51041, 51137, 51257, 51464,
    50059, 50239, 50353, 50481, 50654, 50816, 51042, 51138, 51260, 51465,
    50060, 50240, 50354, 50482, 50656, 50817, 51044, 51139, 51261, 51473,
    50102, 50241, 50355, 50483, 50659, 50818, 51045, 51140, 51262, 51477,
    50103, 50243, 50356, 50485, 50664, 50820, 51046, 51141, 51264, 51479,
    50104, 50245, 50357, 50486, 50665, 50821, 51047, 51142, 51265, 51480,
    50105, 50247, 50358, 50487, 50669, 50822, 51048, 51146, 51266, 51481,
    50106, 50248, 50359, 50488, 50682, 50823, 51049, 51147, 51267, 51482,
    50107, 50249, 50360, 50490, 50683, 50824, 51050, 51148, 51268, 51484,
    50109, 50250, 50361, 50491, 50685, 50952, 51051, 51149, 51269, 51487,
    50111, 50251, 50362, 50492, 50686, 50954, 51052, 51150, 51271, 51488,
    50112, 50252, 50363, 50493, 50687, 50955, 51053, 51151, 51272, 51491,
    50113, 50253, 50364, 50494, 50688, 50956, 51054, 51152, 51273, 51493,
    50114, 50254, 50365, 50496, 50689, 50957, 51055, 51153, 51275, 51556,
    50115, 50255, 50366, 50497, 50690, 50958, 51056, 51154, 51276, 51557,
    50116, 50257, 50367, 50498, 50691, 50959, 51057, 51155, 51277, 51558,
    50117, 50259, 50368, 50499, 50692, 50960, 51058, 51156, 51278, 51559,
    50118, 50260, 50369, 50500, 50693, 50961, 51059, 51159, 51279, 51560,
    50119, 50261, 50370, 50501, 50694, 50962, 51060, 51161, 51280, 51562,
    50121, 50262, 50372, 50502, 50695, 50964, 51061, 51162, 51281, 51563,
    50123, 50263, 50373, 50503, 50696, 50965, 51062, 51163, 51291, 51564,
    50124, 50264, 50374, 50504, 50697, 50966, 51063, 51164, 51292, 51565,
    50125, 50265, 50375, 50507, 50698, 50967, 51064, 51168, 51293, 51566,
    50127, 50266, 50376, 50509, 50699, 50968, 51065, 51169, 51294, 51567,
    50128, 50267, 50377, 50510, 50700, 50969, 51066, 51170, 51295, 51568,
    50129, 50268, 50379, 50514, 50701, 50970, 51067, 51171, 51297, 51569,
    50130, 50269, 50380, 50515, 50702, 50972, 51068, 51173, 51298, 51570,
    50131, 50270, 50381, 50516, 50703, 50973, 51069, 51177, 51299, 51572,
    50132, 50271, 50382, 50518, 50704, 50974, 51070, 51178, 51300, 51573,
    50134, 50272, 50383, 50519, 50705, 50976, 51072, 51179, 51301, 51574,
    50135, 50273, 50385, 50520, 50706, 50977, 51073, 51180, 51302, 51576,
    50142, 50274, 50386, 50521, 50707, 50978, 51074, 51181, 51303, 51577,
    50143, 50275, 50387, 50523, 50708, 50979, 51075, 51182, 51304, 51578,
    50144, 50276, 50388, 50524, 50709, 50981, 51076, 51183, 51305, 51579,
    50145, 50278, 50390, 50525, 50711, 50982, 51077, 51184, 51306, 51580,
    50146, 50282, 50391, 50526, 50722, 50983, 51078, 51185, 51307, 51582,
    50147, 50284, 50397, 50527, 50723, 50984, 51079, 51187, 51308, 51583,
    50148, 50285, 50399, 50528, 50724, 50985, 51080, 51188, 51309, 51584,
    50149, 50287, 50402, 50529, 50725, 50986, 51081, 51189, 51311, 51585,
    50150, 50289, 50403, 50530, 50726, 50987, 51082, 51192, 51313, 51606,
    51607  # see notes below in filter_to_871 for this last subject
]
# fmt: on

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


def filter_to_871(csv: DataFrame) -> DataFrame:
    QC_COLS = [
        "qc_rater_1",
        "qc_anat_rater_2",
        "qc_func_rater_2",
        "qc_anat_rater_3",
        "qc_func_rater_3",
    ]
    qc = csv.loc[:, QC_COLS]
    keep = ~(qc == "fail").any(axis=1)
    csv = csv.loc[keep, :]
    # note there are some discrepancies about whether or not to include subjects:
    #
    #   50279 (Autism) (UM_1)
    #   50286 (Aspergers) (UM_1)
    #   51607 (Aspergers) (MAX_MUN)
    #
    # According to supplementary material of Abraham, they have 46 subjects from MAX_MUN, which we
    # have *if* we include 51607. By contrast, Abraham et al. list only 86 subjects from UM_1, but
    # we have 88 from UM_1 if we include 50279 and 50286. So 51607 is the missing subject in the
    # supplementary files.
    return csv


def download_csv(filter: bool = False) -> DataFrame:
    CSV_FILE = Path(__file__).resolve().parent / "Phenotypic_V1_0b_preprocessed1.csv"
    CSV_WEB_PATH = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
    if not CSV_FILE.exists():
        urlretrieve(CSV_WEB_PATH, CSV_FILE)
    csv = pd.read_csv(CSV_FILE, header=0, na_values="-9999")
    if filter:
        csv = filter_to_871(csv)
    csv = csv.rename(columns={"SUB_ID": "sid", "FILE_ID": "fname"})
    csv.index = csv.sid
    csv.drop(columns="sid", inplace=True)
    return csv


def download_file(url: str, outdir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(f"cd {outdir} && wget --continue {url}", capture_output=True, check=True, shell=True)


def download_rois() -> None:
    start = strftime("%Y-%h-%d--%H:%M")
    LOGFILE = Path(__file__).resolve().parent / f"rois_{start}.log"
    if not LOGFILE.exists():
        LOGFILE.touch()
    csv = download_csv()
    sids = DataFrame(index=pd.Index(SIDS, name="sid", dtype="int64"))
    fids = csv.join(sids, how="inner").fname.to_list()
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
    sids = DataFrame(index=pd.Index(SIDS, name="sid", dtype="int64"))
    fids = csv.join(sids, how="inner").fname.to_list()
    for fid in tqdm(fids, desc=f"Downloading fMRI images"):
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


if __name__ == "__main__":
    download_rois()
