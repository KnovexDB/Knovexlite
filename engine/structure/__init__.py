import torch

from engine.structure.kg.graph import KnowledgeGraph
from engine.structure.kg.index import KGIndex


def get_nbp(name, ckpt, **kwargs):
    if name.lower() == "transe":
        from kg_embedding.nbp_transe import TransE

        nbp = TransE(**kwargs)
    elif name.lower() == "swtranse":
        from kg_embedding.nbp_swtranse import SWTransE

        nbp = SWTransE(**kwargs)
    elif name.lower() == "complex":
        from kg_embedding.nbp_complex import ComplEx

        nbp = ComplEx(**kwargs)
    elif name.lower() == "rotate":
        from kg_embedding.nbp_rotate import RotatE

        nbp = RotatE(**kwargs)
    elif name.lower() == "distmult":
        from kg_embedding.nbp_distmult import DistMult

        nbp = DistMult(**kwargs)
    elif name.lower() == "conve":
        from kg_embedding.nbp_conve import ConvE

        nbp = ConvE(**kwargs)
    elif name.lower() == "rescal":
        from kg_embedding.nbp_rescal import RESCAL

        nbp = RESCAL(**kwargs)
    else:
        raise ValueError(f"Invalid NBP name: {name}")

    if ckpt is not None:
        nbp.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True)
        )

    return nbp
