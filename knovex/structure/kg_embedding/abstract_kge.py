from abc import abstractmethod
from typing import Optional, Union
import logging


import torch

logger = logging.getLogger(__name__)


class KnowledgeGraphEmbedding:
    num_entities: int
    num_relations: int
    device: torch.device

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        This property returns the device of the embeddings.
        """
        pass

    @abstractmethod
    def get_ent_emb(self, ent_id: torch.Tensor) -> torch.Tensor:
        """
        This function returns the embedding for the entity with the given id.
        Inputs:
            ent_id: The tensor of entity ids in the shape [...]
        Returns:
            The tensor of embeddings in the shape [..., embed_dim]
        """
        pass

    @abstractmethod
    def get_rel_emb(self, rel_id: torch.Tensor) -> torch.Tensor:
        """
        This function returns the embedding for the relation with the given id.
        Inputs:
            rel_id: The tensor of relation ids in the shape [...]
        Returns:
            The tensor of embeddings in the shape [..., embed_dim]
        """
        pass

    @abstractmethod
    def embedding_score(
        self,
        head_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        This method computes the score for the triple given the head, tail and
        relation embeddings.

        The score is made into [0, 1].
        The higher score means more likely to be a predicate.

        The computational process is defined by each kge.

        The effect of adaptor is included.

        Assumption:
            The input embeddings are of the same shape
        """
        pass

    @abstractmethod
    def estimate_tail_emb(
        self, head_emb: torch.Tensor, rel_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Given the head and relation embeddings, estimate the tail embedding.
        This is the one-hop inference by LMPNN

        Assumption:
            The input embeddings are of the same shape
        """
        pass

    def id_preproc(self, input_id: Union[torch.Tensor, list]) -> torch.Tensor:
        """
        Handle three cases in input_id
        if input_case is a tensor of the device as self.device,
            return it directly
        if input_case is a list,
            convert it to a tensor of the device as self.device
        if input_case is a tensor of the device in other devices,
            convert it to the device as self.device
        """

        if isinstance(input_id, list):
            input_id = torch.tensor(input_id, dtype=torch.long, device=self.device)
        assert isinstance(input_id, torch.Tensor)
        if input_id.device != self.device:
            input_id = input_id.to(self.device)
        return input_id

    def constraint_score(
        self,
        head_id: torch.Tensor,
        rel_id: torch.Tensor,
        negation: Optional[torch.Tensor] = None,
        tail_id: Optional[torch.Tensor] = None,
    ):
        """
        This function computes the score for the triple given the head, tail
        and relation ids.

        It also cache the results in the __cached__ dictionary to avoid
        redundant computation.

        Inputs:
            - head_id: The tensor of head entity ids in the shape [...]
            - rel_id: The tensor of relation ids in the shape [...]
            - negation: The tensor of negation flags in the shape [...]
            - tail_id:
                1. The tensor of tail entity ids in the shape [...]
                2. If None, the function computes the score for the head and
                    relation against all possible tail entities.

            Note:
                all shapes are broadcastable to [...]

        Returns:
            If tail is not None:
                The tensor of scores in the shape [...] (the broadcasted shape)
            If tail is None:
                The tensor of scores in the shape [..., num_entities]

        Current use case:
        1. used in CQD beam search, where the input is
            - head_id: [num_constraints, beam_size]
            - rel_id: [num_constraints, 1]
            - negation: [num_constraints, 1]
            - tail_id: None
        2. used in fast verifier evaluation and LMBS
            - head_id: [num_constraints, 1]
            - rel_id: [num_constraints, 1]
            - negation: [num_constraints, 1]
            - tail_id: [num_constraints, 1]

        We differenciate those two cases by tail_id is None or not.
        """

        # broadcast everything
        head_id, rel_id = torch.broadcast_tensors(head_id, rel_id)
        if negation is not None:
            negation = torch.broadcast_to(negation, head_id.shape)
        if tail_id is not None:
            tail_id = torch.broadcast_to(tail_id, head_id.shape)
        base_shape = head_id.shape

        if tail_id is None:
            negation = negation.view(base_shape + (1,))

        score = self.embedding_score(head_id, rel_id, tail_id)

        if negation is not None:
            # score = (1 - negation) * score + negation * (1 - score)
            score = torch.where(negation == 1, -score, score)

        return score
