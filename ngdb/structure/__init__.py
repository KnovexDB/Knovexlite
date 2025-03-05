import torch


def get_nbp(name, ckpt, **kwargs):
    if name.lower() == "transe":
        from ngdb.structure.kg_embedding.transe import TransE

        nbp = TransE(**kwargs)
    elif name.lower() == "swtranse":
        from ngdb.structure.kg_embedding.swtranse import SWTransE

        nbp = SWTransE(**kwargs)
    elif name.lower() == "complex":
        from ngdb.structure.kg_embedding.complex import ComplEx

        nbp = ComplEx(**kwargs)
    elif name.lower() == "rotate":
        from ngdb.structure.kg_embedding.rotate import RotatE

        nbp = RotatE(**kwargs)
    elif name.lower() == "distmult":
        from ngdb.structure.kg_embedding.distmult import DistMult

        nbp = DistMult(**kwargs)
    elif name.lower() == "conve":
        from ngdb.structure.kg_embedding.conve import ConvE

        nbp = ConvE(**kwargs)
    elif name.lower() == "rescal":
        from ngdb.structure.kg_embedding.rescal import RESCAL

        nbp = RESCAL(**kwargs)
    else:
        raise ValueError(f"Invalid NBP name: {name}")

    if ckpt is not None:
        nbp.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True)
        )

    return nbp
