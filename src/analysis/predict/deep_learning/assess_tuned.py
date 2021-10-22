# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort:skip
sys.path.append(str(ROOT))  # isort:skip
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

import traceback
from warnings import warn

from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from src.analysis.predict.deep_learning.arguments import get_conv3d_to_lstm3d_config
from src.analysis.predict.deep_learning.callbacks import callbacks
from src.analysis.predict.deep_learning.dataloader import FmriDataset
from src.analysis.predict.deep_learning.models.conv_lstm import Conv3dToConvLstm3d
from src.analysis.predict.deep_learning.tables import tableify_logs

if __name__ == "__main__":
    model_class = Conv3dToConvLstm3d
    seed_everything(333)
    config = get_conv3d_to_lstm3d_config()
    model = model_class(config)
    trainer = Trainer.from_argparse_args(config, callbacks=callbacks(include_optuna=False))
    trainer.logger.log_hyperparams(config)
    train, val = FmriDataset(config).train_val_split(config)
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val,
        batch_size=config.val_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        drop_last=False,
    )
    trainer.fit(model, train_loader, val_loader)
    try:
        df = tableify_logs(trainer)
        print(df)
    except Exception:
        traceback.print_exc()
        warn("No data logged to dataframes")
