from typing import Optional
import logging
import torch
from torch import nn

from .abstract_kge import KnowledgeGraphEmbedding

logger = logging.getLogger(__name__)


class ComplEx(KnowledgeGraphEmbedding, nn.Module):

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        max_eval_flux: int = 100000,  # eval at most this many triples each time
        **kwargs,  # for compatibility
    ):
        super(ComplEx, self).__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.max_eval_flux = max_eval_flux

        self._entity_embedding = nn.Embedding(num_entities, 2 * embedding_dim)
        self._relation_embedding = nn.Embedding(
            num_relations, 2 * embedding_dim
        )

    @property
    def device(self):
        return self._entity_embedding.weight.device

    def get_ent_emb(self, ent_id: torch.Tensor):
        return self._entity_embedding(self.id_preproc(ent_id))

    def get_rel_emb(self, rel_id: torch.Tensor):
        return self._relation_embedding(self.id_preproc(rel_id))

    def embedding_score(
        self,
        head_id: torch.Tensor,
        rel_id: torch.Tensor,
        tail_id: Optional[torch.Tensor] = None,
    ):
        if tail_id is None:
            head_emb = self.get_ent_emb(head_id)
            rel_emb = self.get_rel_emb(rel_id)
            est_tail = self.estimate_tail_emb(head_emb, rel_emb)
            head_emb = head_emb.expand_as(est_tail)
            rel_emb = rel_emb.expand_as(est_tail)
            if self.vector_adaptor is not None:
                est_tail = self.vector_adaptor(head_emb, rel_emb, est_tail)
            score_list = []
            begin = 0
            est_tail = est_tail.unsqueeze(-2)  # [..., 1, emb_dim]
            head_emb = head_emb.unsqueeze(-2)  # [..., 1, emb_dim]
            rel_emb = rel_emb.unsqueeze(-2)  # [..., 1, emb_dim]
            num_head_rel = torch.prod(torch.tensor(head_emb.shape[:-1])).item()
            adaptive_batch_size = self.max_eval_flux // num_head_rel + 1

            while begin < self.num_entities:
                end = min(begin + adaptive_batch_size, self.num_entities)
                emb = self.get_ent_emb(torch.arange(begin, end))
                # make the est_tail and emb in broadcastable shape
                score = self.entity_pair_scoring(est_tail, emb)
                score_list.append(score)
                begin = end
                score = torch.cat(score_list, dim=-1)

            if self.score_adaptor is not None:
                score = self.score_adaptor(head_emb, rel_emb, score)
        else:
            num_triples = head_id.shape[0]
            begin = 0
            while begin < num_triples:
                end = min(begin + self.max_eval_flux, num_triples)
                head_emb = self.get_ent_emb(head_id[begin:end])
                rel_emb = self.get_rel_emb(rel_id[begin:end])
                est_tail = self.estimate_tail_emb(head_emb, rel_emb)
                if self.vector_adaptor is not None:
                    est_tail = self.vector_adaptor(head_emb, rel_emb, est_tail)
                _score = self.entity_pair_scoring(
                    est_tail, self.get_ent_emb(tail_id[begin:end])
                )
                if self.score_adaptor is not None:
                    _score = self.score_adaptor(head_emb, rel_emb, _score)
                if begin == 0:
                    score = _score
                else:
                    score = torch.cat([score, _score], dim=0)
                begin = end

        return score

    def estimate_tail_emb(self, head_emb, rel_emb):
        lhs = (
            head_emb[..., : self.embedding_dim],
            head_emb[..., self.embedding_dim :],
        )
        rel = (
            rel_emb[..., : self.embedding_dim],
            rel_emb[..., self.embedding_dim :],
        )

        return torch.cat(
            [
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0],
            ],
            -1,
        )

    def entity_pair_scoring(self, emb1, emb2):
        """
        This function computes the score for the pair of entity embeddings.
        """
        scores = torch.sum(emb1 * emb2, dim=-1)
        return scores

    def regularization(self, emb):
        r, i = emb[..., : self.embedding_dim], emb[..., self.embedding_dim :]
        norm_vec = torch.sqrt(r**2 + i**2)
        reg = torch.sum(norm_vec**3, -1)
        return reg
