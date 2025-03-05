from collections import defaultdict
from typing import List, Tuple, NewType

import torch
from torch_geometric.data import Data

from ngdb.structure.knowledge_graph_index import KGIndex
from ngdb.utils.data import iter_triple_from_tsv

Triple = NewType("Triple", Tuple[int, int, int])


class KnowledgeGraph:
    """
    Fully tensorized
    """

    def __init__(
        self, triples: List[Triple], kgindex: KGIndex, device="cpu", **kwargs
    ):

        self.triples = triples
        self.kgindex = kgindex

        self.num_entities = kgindex.num_entities
        self.num_relations = kgindex.num_relations

        self.device = device

        self.hr2t = defaultdict(set)
        self.tr2h = defaultdict(set)
        self.r2ht = defaultdict(set)
        self.ht2r = defaultdict(set)
        self.h2r = defaultdict(set)
        self.r2h = defaultdict(set)
        self.r2t = defaultdict(set)
        self.h2t = defaultdict(set)
        self.t2h = defaultdict(set)

        self.triple_set = set(triples)

        for h, r, t in self.triples:
            self.hr2t[(h, r)].add(t)
            self.tr2h[(t, r)].add(h)
            self.r2ht[r].add((h, t))
            self.ht2r[(h, t)].add(r)
            self.h2r[h].add(r)
            self.h2t[h].add(t)
            self.t2h[t].add(h)
            self.r2h[r].add(h)
            self.r2t[r].add(t)

        # build pytorch tensor storage
        self.triple_tensor = torch.tensor(
            self.triples, dtype=torch.long, device=self.device
        )

        self.edge_head = self.triple_tensor[:, 0]
        self.edge_attr = self.triple_tensor[:, 1]
        self.edge_tail = self.triple_tensor[:, 2]

        self.edge_index = torch.cat(
            [self.edge_head.view(1, -1), self.edge_tail.view(1, -1)], dim=0
        )

        self.h2r_mask = torch.zeros(
            self.num_entities, self.num_relations, dtype=torch.bool, device=self.device
        )

        for h in self.h2r:
            self.h2r_mask[h, list(self.h2r[h])] = True

    @property
    def build_pyg_data(self):
        return Data(
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_entities,
        )

    @classmethod
    def create(cls, triple_files, kgindex: KGIndex, **kwargs):
        """
        Create the class
        TO be modified when certain parameters controls the triple_file
        triple files can be a list
        """
        triples = []
        for h, r, t in iter_triple_from_tsv(triple_files):
            assert h in kgindex.inverse_entity_id_to_name
            assert r in kgindex.inverse_relation_id_to_name
            assert t in kgindex.inverse_entity_id_to_name
            triples.append((h, r, t))

        return cls(triples, kgindex=kgindex, **kwargs)
