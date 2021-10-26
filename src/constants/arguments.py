import os
from argparse import Namespace
from typing import Dict

ARG_DEFAULTS = Namespace(
    **dict(
        batch_size=4,
        val_batch_size=4,
        num_workers=8,
    )
)
# overrides to some lightning default arg values
OVERRIDE_DEFAULTS: Dict = (
    dict(
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html?highlight=logging#logging-frequency
        log_every_n_steps=5,
        # flush_logs_every_n_steps=20,
        max_time={"hours": 2},
    )
    if os.environ.get("CC_CLUSTER") is None
    else dict(max_time={"hours": 2})
)
PBAR = 0 if os.environ.get("CC_CLUSTER") is not None else 1
