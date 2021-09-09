from pathlib import Path

import optuna

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    dbs = sorted(ROOT.rglob("*db"))
    for db in dbs:
        study_name = db.stem
        storage = f"sqlite:///{db}"
        study = optuna.study.load_study(study_name, storage)
        df = study.trials_dataframe()
        outfile = ROOT / f"{study_name}.json"
        df.to_json(outfile)
        print(f"Saved study in {db} to {outfile}")
