from . import (
    approximators,
    adapters,
    datasets,
    diagnostics,
    distributions,
    experimental,
    networks,
    simulators,
    workflows,
    utils,
)

from .adapters import Adapter
from .approximators import ContinuousApproximator, PointApproximator
from .datasets import OfflineDataset, OnlineDataset, DiskDataset
from .simulators import make_simulator
from .workflows import BasicWorkflow


def setup():
    # perform any necessary setup without polluting the namespace
    import keras
    import logging

    # set the basic logging level if the user hasn't already
    logging.basicConfig(level=logging.INFO)

    # use a separate logger for the bayesflow package
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    from bayesflow.utils import logging

    logging.debug(f"Using backend {keras.backend.backend()!r}")

    if keras.backend.backend() == "torch":
        import torch

        torch.autograd.set_grad_enabled(False)

        logging.warning(
            "Autograd is disabled by default to avoid excessive memory usage. "
            "If you need gradients (e.g., custom training loops), use\n"
            "with torch.enable_grad():"
        )


# call and clean up namespace
setup()
del setup
