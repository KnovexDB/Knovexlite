from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Batch

from ngdb.structure.kg_embedding.abstract_kge import (
    KnowledgeGraphEmbedding as KGE,
)



class Reasoner(ABC):
    """
    An abstrcat interface for a reasoner.
    """

    @abstractmethod
    def set_nbp(self, kge: KGE):
        """
        Set the nbp for the reasoner.
        """
        pass

    @abstractmethod
    def train_loss(self, batch: Batch, target: torch.Tensor):
        """
        Compute the training loss.
        """
        pass

    @abstractmethod
    def eval_all_entity_scores(self, batch: Batch):
        """
        Evaluate the scores of all entities.
        """
        pass
