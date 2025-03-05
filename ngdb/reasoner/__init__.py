import logging


def get_reasoner(name, **kwargs):
    logger = logging.getLogger(__name__)

    if name.lower() == "lmpnn":
        from engine.reasoner.lmpnn import LMPNN
        reasoner = LMPNN(**kwargs)
    elif name.lower() == "cqd":
        from engine.reasoner.cqd import CQDBeam
        reasoner = CQDBeam(**kwargs)
    else:
        raise ValueError("Invalid reasoner name")

    logger.info(f"Loading reasoner {name}")
    logger.info(f"Loading config {kwargs}")
    return reasoner
