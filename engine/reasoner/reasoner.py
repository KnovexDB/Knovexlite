from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn
from torch_geometric.data import Batch

from engine.structure.kg_embedding.kge_interface import (
    KnowledgeGraphEmbedding as KGE,
)


class MLP(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_size, num_hidden_layers
    ):
        super(MLP, self).__init__()
        layers = []
        in_channels = in_channels
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_channels, hidden_size))
            layers.append(nn.ReLU())
            in_channels = hidden_size
        layers.append(nn.Linear(hidden_size, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Reasoner(ABC):
    """
    An abstrcat interface for a reasoner.
    """

    @abstractmethod
    def set_nbp(self, nbp: NBP):
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
