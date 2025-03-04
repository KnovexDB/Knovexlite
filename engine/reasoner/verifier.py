import logging
from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter, index_to_mask
from tqdm import trange

from src.structure.neural_binary_predicate import NeuralBinaryPredicate as NBP
from src.structure.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class Verifier(nn.Module):
    """
    A verifier has the following fields

    1. A neural binary predicate
    2. conj_aggr, a str in ['min', 'mul']
    3. random_negative_edges, a float in [0, 1] to determine the prob. of
    random negative edges

    and the following methods

    1. set_nbp(self, nbp): set the nbp by external input
    2. forward(self, batch): forward the batch (of conjunctive query graphs)
    and finally calculates the truth value of each graph.
    3. train_loss(self, batch, target): compute the training loss


    The key difference that distinguish a PyG query graph and a subgraph is that
    in query graph, we use .x to distinguish the entities and quantifiers.
    In subgraph, we use edge_index directly.
    """

    def __init__(
        self,
        conj_aggr: str = "min",
        nbp: "Optional[NBP]" = None,
        kg: "Optional[KnowledgeGraph]" = None,
    ) -> None:
        super().__init__()
        self.conj_aggr: str = conj_aggr
        self.nbp: "Optional[NBP]" = nbp
        self.kg = kg

    def set_nbp(self, nbp: "NBP") -> None:
        self.nbp = nbp

    def forward(self, batch: "Batch") -> "torch.Tensor":
        """
        Get all truth values of each edge
        """
        head: "torch.Tensor" = batch.x[batch.edge_index[0]]
        relation: "torch.Tensor" = batch.edge_attr[:, 0]
        tail: "torch.Tensor" = batch.x[batch.edge_index[1]]
        negation: "torch.Tensor" = batch.edge_attr[:, 1]

        score: "torch.Tensor" = self.nbp.constraint_score(
            head, relation, negation, tail
        )
        return score

    def evaluate_fuzzy_truth_value(self, batch: "Batch") -> "torch.Tensor":
        """
        Evaluate the fuzzy truth value of the conjunctive query graph
        """

        batch_by_edge = batch.batch[batch.edge_index[0]]
        score: "torch.Tensor" = self.forward(batch)
        return scatter(
            src=score,
            index=batch_by_edge,
            dim=0,
            dim_size=batch.num_graphs,
            reduce=self.conj_aggr,
        )

    def train_loss(self, batch: "Batch") -> "torch.Tensor":
        output: "torch.Tensor" = self.evaluate_fuzzy_truth_value(batch)
        assert not torch.isnan(output).any()
        loss = -torch.log(output.sigmoid() + 1e-10).mean()
        assert not torch.isnan(loss).any()
        return loss


class AEFGVerifier(nn.Module):
    """
    This verifier modifies the Verifier class
    with a pre-defined play of AEFG.
    """

    def __init__(
        self,
        nbp: "Optional[NBP]" = None,
        kg: "Optional[KnowledgeGraph]" = None,
        round: int = 3,
    ) -> None:
        super().__init__()
        self.nbp: "Optional[NBP]" = nbp
        self.kg = kg
        self.round = round

        self._cached_lowest_score_h2srt = defaultdict(list)
        self._cached_lowest_score_ht2sr = defaultdict(list)

    def set_nbp(self, nbp: "NBP") -> None:
        self.nbp = nbp

    def train_loss(self, ent_ten: torch.Tensor):
        """
        input is the entity tensor
        """
        batch = self.play(ent_ten)
        target_score = self.evaluate_fuzzy_truth_value(batch)
        target_score -= target_score.max(dim=1, keepdim=True).values.detach()

        log_denominator = torch.logsumexp(target_score, dim=1)
        positive_mask = batch.target
        positive_score = target_score[positive_mask]
        target_idx = torch.arange(
            target_score.shape[0], device=target_score.device
        ).repeat_interleave(positive_mask.sum(-1, keepdim=False))
        logger.info(target_idx)
        logger.info(log_denominator.size())
        log_numerator = torch.scatter_reduce(
            input=torch.zeros_like(log_denominator),
            dim=0,
            index=target_idx,
            src=positive_score,
            reduce="mean",
            include_self=False,
        )

        logger.info(f"Log numerator: {log_numerator}")
        logger.info(f"Log denominator: {log_denominator}")

        loss = -log_numerator.mean() + log_denominator.mean()
        return loss

    def evaluate_fuzzy_truth_value(self, batch: Batch) -> "torch.Tensor":
        """
        Evaluate the fuzzy truth value of the conjunctive query graph
        """
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        head_id = batch.x[edge_index[0], 0]
        rel_id = edge_attr[:, 0]
        neg = edge_attr[:, 1]

        score = self.nbp.constraint_score(head_id, rel_id, neg)
        num_tails = score.size(1)
        assert num_tails == self.nbp.num_entities

        inverse_index = torch.unique(edge_index[1], return_inverse=True)[1]

        target_score = torch.scatter_reduce(
            input=torch.zeros(
                batch.num_graphs, num_tails, device=head_id.device
            ),
            dim=0,
            index=inverse_index.view(-1, 1).expand(-1, num_tails),
            src=score,
            reduce="sum",
            include_self=False,
        )
        return target_score

    def cache_scores(self):
        triple = self.kg.triple_tensor.T
        head_id, rel_id, tail_id = self.kg.triple_tensor.T
        with torch.no_grad():
            scores = self.nbp.constraint_score(
                head_id, rel_id, tail_id=tail_id
            )

        for i in trange(self.kg.triples):
            h, r, t = triple[:, i]
            s = scores[i]
            self._cached_lowest_score_h2srt[h].append((s, r, t))
            self._cached_lowest_score_ht2sr[(h, t)].append((s, r))

        # calculate the offset
        self.offset = torch.tensor(
            [0] + torch.bincount(self.triple[0]).cumsum(0).tolist(),
            dtype=torch.long,
            device=self.triple.device,
        )

    # def _find_hardest_rel_tail(self, head_ten: torch.Tensor, shift: int = 0):
    #     """
    #     Find the lowest relation for a given head and tail
    #     """
    #     # becuase the quaduple is sorted by score, tail, rel, head
    #     lo_rel = self.triple[1, self.offset[head_ten] + shift]
    #     lo_tail = self.triple[2, self.offset[head_ten] + shift]
    #     lo_score = self.scores[self.offset[head_ten] + shift]
    #     return lo_rel, lo_tail, lo_score

    def _play_one_round(self, head: torch.Tensor) -> torch.Tensor:
        """
        It should suggest relations and tails for the given heads

        Input:
            - head [batch_size, rounds]
        Return:
            - head [batch_size, rounds + 1]
            - rel [batch_size, rounds]
            - negation [batch_size, 1]
        """
        batch_size, rounds = head.shape

        hard_head = head[:, -1]

        hard_rel, hard_tail, hard_score = self._find_hardest_rel_tail(
            hard_head, shift=rounds - 1
        )

        if rounds == 1:
            new_head = torch.cat([hard_head, hard_tail], dim=1)
            return new_head, hard_rel, torch.ones_like(hard_rel)

        # identify the relations

        head_rest = head[:, :-1]
        tail_rest = hard_tail.repeat(1, rounds - 1)
        rel_all = torch.arange(self.kg.num_relations, device=head.device)

        # get the scores
        scores = self.nbp.constraint_score(
            head_id=head_rest.view(-1, 1),
            rel_id=rel_all.view(1, -1),
            tail_id=tail_rest.view(-1, 1),
        )
        # find the best relation by the highest score
        easy_rel_flat = scores.argmax(dim=1).view(-1)
        easy_rel = easy_rel_flat.view(batch_size, rounds - 1)

        # check the observation of head_rest, tail_rest, and easy_rel
        head_rest_flat = head_rest.view(-1)
        tail_rest_flat = tail_rest.view(-1)
        negation_flat = torch.tensor(
            [
                tail_rest_flat[i]
                not in self.kg.hr2t[(head_rest_flat[i], easy_rel_flat[i])]
                for i in range(head_rest_flat.shape[0])
            ],
            device=head.device,
        )
        negation = negation_flat.view(batch_size, rounds - 1)

        # easy_hard_rel
        reordered_head = torch.cat([head, hard_tail], dim=1)
        easy_hard_rel = torch.cat([easy_rel, hard_rel], dim=1)
        negation = torch.cat([negation, torch.zeros_like(hard_rel)], dim=1)
        return reordered_head, easy_hard_rel, negation

    def play(self, batch_ent_ten: torch.Tensor) -> "torch.Tensor":
        """
        Input:
            - batch_ent_ten [batch_size, num_ent]
        """
        device = batch_ent_ten.device
        batch_size = batch_ent_ten.shape[0]
        head_ = batch_ent_ten.view(-1, 1)

        data_list = []
        for r in range(1, self.round + 1):
            new_head_, rel_, neg_ = self._play_one_round(head_)

            for i in range(batch_size):
                head, rel, neg = head_[i], rel_[i], neg_[i]
                x = head.new_zeros(r + 1, 2)
                x[:r, 0] = head
                x[r, 1] = 2
                edge_index = torch.stack(
                    [
                        torch.arange(r, device=device),
                        torch.ones_like(head, dtype=torch.long) * r,
                    ],
                    dim=0,
                )
                edge_attr = torch.cat(
                    [rel.view(-1, 1), neg.view(-1, 1)], dim=1
                )

                tail_mask = torch.ones(
                    1, self.kg.num_entities, dtype=torch.bool, device=device
                )

                num_heads = head.shape[0]
                for j in range(num_heads):
                    tail = self.kg.hr2t[head[j], rel[j]]
                    _t_mask = index_to_mask(
                        torch.tensor(
                            list(tail), device=device, dtype=torch.long
                        ),
                        size=self.kg.num_entities,
                    )
                    if neg[j]:
                        _t_mask = ~_t_mask
                    tail_mask = tail_mask & _t_mask

                assert tail_mask.sum() > 0

                # add the new form new query graphs
                query_graph = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    target=tail_mask,
                )

                data_list.append(query_graph)

            head_ = new_head_

        batch = Batch.from_data_list(data_list, follow_batch=["target"])
        return batch
