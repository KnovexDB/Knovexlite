from collections import defaultdict
from typing import List, Tuple, NewType

import torch
import numpy as np
from torch_geometric.data import Data

from knovex.structure.kg.index import KGIndex
from knovex.utils.data import iter_triple_from_tsv
from knovex.utils.class_util import fixed_depth_nested_dict


Triple = NewType("Triple", Tuple[int, int, int])


class KnowledgeGraph:
    """
    Fully tensorized
    """

    def __init__(
        self, triples: List[Triple], kgindex: KGIndex,  tensorize=False, device="cpu", **kwargs
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
        self.node2or = fixed_depth_nested_dict(int, 2)
        self.node2ir = fixed_depth_nested_dict(int, 2)
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
            self.node2or[h][r] += 1
            self.node2ir[t][r] += 1

        # build pytorch tensor storage
        if tensorize:
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

def kg2matrix(kg: KnowledgeGraph, use_connected_part: bool = False):
    """
    The nodes of kg should always be labeled by number first.
    use_connected_part: if True, only the connected part of the graph will be used.
    """
    if use_connected_part:
        all_node_list = set(kg.node2or.keys()).union(set(kg.node2ir.keys()))
        node_num = len(all_node_list)
    else:
        node_num = kg.num_entities
    kg_matrix = np.zeros((node_num, node_num), dtype=int)
    for triple in kg.triples:
        head, relation, tail = triple
        kg_matrix[head][tail] += 1
    return kg_matrix

def labeling_triples(triple_list):
    labeled_nodes = set()
    node2index = {}
    index2node = {}
    now_index = 0
    new_triple_list = []
    for triple in triple_list:
        head, relation, tail = triple
        if head not in labeled_nodes:
            labeled_nodes.add(head)
            node2index[head] = now_index
            index2node[now_index] = head
            now_index += 1
        if tail not in labeled_nodes:
            labeled_nodes.add(tail)
            node2index[tail] = now_index
            index2node[now_index] = tail
            now_index += 1
        translated_triples = (node2index[head], relation, node2index[tail])
        new_triple_list.append(translated_triples)
    return new_triple_list, node2index, index2node
