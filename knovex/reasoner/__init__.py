import logging


def get_reasoner(name, **kwargs):
    logger = logging.getLogger(__name__)

    if name.lower() == "lmpnn":
        from knovex.reasoner.lmpnn import LMPNN
        reasoner = LMPNN(**kwargs)
    elif name.lower() == "cqd":
        from knovex.reasoner.cqd import CQDBeam
        reasoner = CQDBeam(**kwargs)
    elif name.lower() == "fit":
        from knovex.reasoner.fit import FITReasoner
        reasoner = FITReasoner(**kwargs)
    else:
        raise ValueError("Invalid reasoner name")

    logger.info(f"Loading reasoner {name}")
    logger.info(f"Loading config {kwargs}")
    return reasoner
