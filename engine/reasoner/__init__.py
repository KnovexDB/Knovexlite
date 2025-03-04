import logging

import torch


def get_reasoner(name, **kwargs):
    logger = logging.getLogger(__name__)

    if name.lower() == "lmpnn":
        from src.reasoner.lmpnn import LMPNN
        reasoner = LMPNN(**kwargs)
    elif name.lower() == "cqd":
        from src.reasoner.cqd import CQDBeam
        reasoner = CQDBeam(**kwargs)
    elif name.lower() == "fit":
        from src.reasoner.fit import FITBeam
        reasoner = FITBeam(**kwargs)
    elif name.lower() == "lmbs":
        from src.reasoner.lmbs import LMBS
        reasoner = LMBS(**kwargs)
    else:
        raise ValueError("Invalid reasoner name")

    logger.info(f"Loading reasoner {name}")
    logger.info(f"Loading config {kwargs}")
    return reasoner
