import heapq
import random
from collections import defaultdict
from typing import List
import logging

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.neural_binary_predicate import NeuralBinaryPredicate as NBP
from src.utils.data import VariadicMatrix

class AEFGPyGDataset:
    def __init__(
        self,
        kg: KnowledgeGraph,
        nbp: NBP,
        round: int = 3,
        negative_tails_to_cache: int = 32,
        preturb: float = 0.0,
        conj_aggr: str = "mul"
    ):
        self.kg: KnowledgeGraph = kg
        self.nbp: NBP = nbp
        self.round: int = round
        self.negative_tails_to_cache: int = negative_tails_to_cache
        self.preturb: float = preturb
        self.conj_aggr = conj_aggr

        self._cached_lowest_score_h2srt = defaultdict(list)
        self._cached_lowest_score_ht2sr = defaultdict(list)

        self._cached_lowest_neg_score_h2srt = defaultdict(list)
        self._cached_lowest_neg_score_ht2sr = defaultdict(list)

        self.quaduple = None

        self._dataset = []

    def _cache_scores(self):
        """
        Cache all the scores for the entities
        """
        # clear the cache
        self._cached_lowest_score_h2srt = defaultdict(list)
        self._cached_lowest_score_ht2sr = defaultdict(list)

        self._cached_lowest_neg_score_h2srt = defaultdict(list)
        self._cached_lowest_neg_score_ht2sr = defaultdict(list)

        # it stores the scores of the least relation and score
        # key: head
        # value: a list
        # each element in the list (score, relation, tail) of lowest scores
        # the length of the list is the number of rounds self.round
        # self._cached_scores is constructed by heapq

        # get triple dataloader
        dataloader = DataLoader(
            dataset=self.kg.triple_tensor,
            batch_size=10000,
        )
        for batch in tqdm(dataloader, desc="Caching positive scores"):
            batch = batch.to(self.nbp.device)
            head, rel, tail = batch[:, 0], batch[:, 1], batch[:, 2]
            head_emb = self.nbp.get_ent_emb(head)
            rel_emb = self.nbp.get_rel_emb(rel)
            tail_emb = self.nbp.get_ent_emb(tail)

            score = self.nbp.embedding_score(head_emb, rel_emb, tail_emb)

            for i in range(len(head)):
                h = head[i].item()
                r = rel[i].item()
                t = tail[i].item()
                s = score[i].item()

                self._cached_lowest_score_h2srt[h].append((s, r, t))
                self._cached_lowest_score_ht2sr[(h, t)].append((s, r))

            neg_tail = torch.randint(
                low=0,
                high=self.kg.num_entities,
                size=(len(head), self.negative_tails_to_cache),
                device=self.nbp.device,
            )
            neg_score = self.nbp.embedding_score(
                head_emb.unsqueeze(1),
                rel_emb.unsqueeze(1),
                self.nbp.get_ent_emb(neg_tail),
            )

            for i in range(len(head)):
                h = head[i].item()
                r = rel[i].item()
                for j in range(neg_tail.size(1)):
                    t = neg_tail[i, j].item()
                    s = neg_score[i, j].item()
                    self._cached_lowest_neg_score_h2srt[h].append((s, r, t))
                    self._cached_lowest_neg_score_ht2sr[(h, t)].append((s, r))

        for h in self._cached_lowest_score_h2srt:
            self._cached_lowest_score_h2srt[h] = heapq.nsmallest(
                self.round,
                self._cached_lowest_score_h2srt[h],
                key=lambda x: x[0],
            )

        for ht in self._cached_lowest_score_ht2sr:
            self._cached_lowest_score_ht2sr[ht] = heapq.nsmallest(
                self.round,
                self._cached_lowest_score_ht2sr[ht],
                key=lambda x: x[0],
            )

        for h in self._cached_lowest_neg_score_h2srt:
            self._cached_lowest_neg_score_h2srt[h] = heapq.nsmallest(
                self.round,
                self._cached_lowest_neg_score_h2srt[h],
                key=lambda x: x[0],
            )

        for ht in self._cached_lowest_neg_score_ht2sr:
            self._cached_lowest_neg_score_ht2sr[ht] = heapq.nsmallest(
                self.round,
                self._cached_lowest_neg_score_ht2sr[ht],
                key=lambda x: x[0],
            )


    def play(self, entity_list: List[int]):
        """
        Given an entity, play a EFGame with self.cache_scores.

        The game is played in the following way:

        1. Given an entity list, for each entity h in the entity list, we
        retrieve the (s, r, t) from self.cache_scores where s is the lowest
        score of triples starting with h.
        2. We compare all retrieved scores and greedily choose the lowest score
        Also, pay attention to that we want new entity t, which is not included
        in the entity list.
        """

        if len(self._cached_lowest_score_h2srt) == 0:
            self._cache_scores()

        # find the lowest score from a list of heapified scores, with the
        # constraint that the tail is not in the entity list.
        # if no such entity is found, return None

        # list of heapified scores: self._cached_lowest_score_h2srt[h]
        new_entity, min_score = None, float("inf")
        for i in range(self.round):
            for h in entity_list:
                if h in self._cached_lowest_score_h2srt:
                    if len(self._cached_lowest_score_h2srt[h]) > i:
                        s, r, t = self._cached_lowest_score_h2srt[h][i]
                    if len(self._cached_lowest_neg_score_h2srt[h]) > i:
                        neg_s, neg_r, neg_t = (
                            self._cached_lowest_neg_score_h2srt[h][i]
                        )
                        if neg_s < s:
                            s, r, t = neg_s, neg_r, neg_t

                    if (
                        new_entity is None or s < min_score
                    ) and t not in entity_list:
                        new_entity, r, min_score = t, r, s
            if new_entity is not None:
                break

        # add the new entity to the entity list
        return new_entity, r

    def __len__(self):
        return self.kg.num_entities

    def __getitem__(self, entity_id):
        return self._process_v2(entity_id)

    def _process_v2(self, entity_id):
        edge_index_list = []
        edge_attr_list = []
        entity_id_to_node_id = {entity_id: 0}
        node_id_to_entity_id = {0: entity_id}

        entity_list = [entity_id]
        for i in range(self.round):
            new_entity, r = self.play(entity_list)
            if new_entity is None:
                break

            # when new entity is added, construct the pairwise edges
            node_id = len(entity_id_to_node_id)
            entity_id_to_node_id[new_entity] = node_id
            node_id_to_entity_id[node_id] = new_entity

            # construct the pairwise edges of new_entity to old_entity_list
            # now you have a new entity list, derived by EFG
            for old_entity in entity_list:
                # construct the pairwise edges of old_entity to new_entity
                old_node_id = entity_id_to_node_id[old_entity]
                new_node_id = entity_id_to_node_id[new_entity]

                edge_index_list.append([old_node_id, new_node_id])

                # chech the relation and negation
                pair = (old_entity, new_entity)
                neg = 0
                if pair in self._cached_lowest_score_ht2sr:
                    s, r = self._cached_lowest_score_ht2sr[pair][0]
                elif pair in self._cached_lowest_neg_score_ht2sr:
                    s, r = self._cached_lowest_neg_score_ht2sr[pair][0]
                    neg = 1
                else:
                    neg = 1

                edge_attr_list.append([r, neg])

            entity_list.append(new_entity)

        # construct the PyG data object
        x = torch.tensor(
            [
                node_id_to_entity_id[node_id]
                for node_id in node_id_to_entity_id
            ],
            dtype=torch.long,
            device=self.nbp.device,
        )
        edge_index = torch.tensor(
            edge_index_list, dtype=torch.long, device=self.nbp.device
        ).T
        edge_attr = torch.tensor(
            edge_attr_list, dtype=torch.long, device=self.nbp.device
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(node_id_to_entity_id),
        )
