import logging
from typing import Optional, List
from random import choice

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
import torch
from torch import nn
from torch.nn import functional as F


from engine.structure.kg_embedding.kge_interface import (
    KnowledgeGraphEmbedding as KGE,
)
from engine.reasoner.reasoner import Reasoner

logger = logging.getLogger(__name__)


class LMPLayerBiasOnly(MessagePassing):
    def __init__(
        self, embedding_dim, hidden_size, num_hidden_layers, nbp: Optional[NBP]
    ):
        super(LMPLayerBiasOnly, self).__init__(aggr="add")
        self.nbp: NBP = nbp
        self.bias: nn.Parameter = None
        self.scale: nn.Parameter = None
        self.entity_emb = None

    def update_net(self, x):
        """
        x.shape = [batch_size, embedding_dim]
        """
        entity_score = x.matmul(self.entity_emb.t())
        entity_score = entity_score * self.scale + self.bias
        entity_score.relu_()
        out = entity_score.matmul(self.entity_emb)
        return out

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        relation_id, neg_flag = edge_attr[:, 0], edge_attr[:, 1]
        relation_embedding = self.nbp.get_rel_emb(relation_id)
        tail_pred = self.nbp.estimate_tail_emb(x_j, relation_embedding)

        neg_coef = 1 - 2 * neg_flag.float()

        tail_pred_with_neg = tail_pred * neg_coef.view(-1, 1)

        return tail_pred_with_neg

    def update(self, aggr_out, x):
        # this is different from the original implementation
        # we further use layer normalization
        aggr = 0.1 * x + aggr_out
        transformed = self.update_net(aggr)

        return transformed


class LMPLayer(MessagePassing):
    def __init__(
        self, embedding_dim, hidden_size, num_hidden_layers, nbp: Optional[NBP]
    ):
        super(LMPLayer, self).__init__(aggr="add")
        self.nbp: NBP = nbp
        self.update_net = MLP(
            embedding_dim, embedding_dim, hidden_size, num_hidden_layers
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        relation_id, neg_flag = edge_attr[:, 0], edge_attr[:, 1]
        relation_embedding = self.nbp.get_rel_emb(relation_id)
        tail_pred = self.nbp.estimate_tail_emb(x_j, relation_embedding)

        neg_coef = 1 - 2 * neg_flag.float()

        tail_pred_with_neg = tail_pred * neg_coef.view(-1, 1)

        return tail_pred_with_neg

    def update(self, aggr_out, x):
        # this is different from the original implementation
        # we further use layer normalization
        aggr = 0.1 * x + aggr_out
        transformed = self.update_net(aggr)

        return transformed


class LMPNN(nn.Module, Reasoner):
    def __init__(
        self,
        hidden_size,
        num_hidden_layers,
        embedding_dim,
        negative_sample_size,
        T,
        loss="softmax",
        bias_only=True,
        **kwargs,
    ):
        super(LMPNN, self).__init__()
        self.nbp: Optional[NBP] = None
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embedding_dim = embedding_dim
        self.negative_sample_size = negative_sample_size
        self.T = T

        self.free_variable_emb = nn.Parameter(torch.randn(embedding_dim))
        self.existential_variable_emb = self.free_variable_emb

        if bias_only:
            self.lmp = LMPLayerBiasOnly(
                embedding_dim, hidden_size, num_hidden_layers, self.nbp
            )
        else:
            self.lmp = LMPLayer(
                embedding_dim, hidden_size, num_hidden_layers, self.nbp
            )
        self.loss = loss

    def set_nbp(self, nbp: NBP):
        self.nbp = nbp
        self.lmp.nbp = nbp
        if hasattr(self.lmp, "bias"):
            self.lmp.bias = nn.Parameter(
                torch.zeros(nbp.num_entities, device=nbp.device)
            )
        if hasattr(self.lmp, "scale"):
            self.lmp.scale = nn.Parameter(
                torch.ones(nbp.num_entities, device=nbp.device)
            )
        if hasattr(self.lmp, "entity_emb"):
            self.lmp.entity_emb = self.nbp._entity_embedding.weight

    def forward(self, batch: Batch):
        """
        Input pyg graph and output the embeddings of the free variables.
        """
        max_num_vars = batch.num_vars.max().item()
        # get the embeddings of all entities
        x = self.nbp.get_ent_emb(batch.x[:, 0])
        x[batch.x[:, 1] == 1] = self.existential_variable_emb
        x[batch.x[:, 1] == 2] = self.free_variable_emb

        # get the embeddings of all relations
        x_by_layers = []
        for i in range(max_num_vars):
            x = self.lmp(
                x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
            )
            x_by_layers.append(x)

        x_stacks = torch.stack(
            x_by_layers, dim=1
        )  # [all nodes, layers, embedding_dim]
        logger.debug(f"x_stacks: {x_stacks.size()}")
        # get the free variable embeddings
        # you need to filter out all indexes of free nodes
        # then you need to get the layer of num_vars

        free_variable_idx = torch.nonzero(batch.x[:, 1] == 2).squeeze()
        logger.debug(f"free_variable_idx: {free_variable_idx.size()}")
        free_var_emb = x_stacks[free_variable_idx]
        logger.debug(f"free_var_emb: {free_var_emb.size()}")

        free_variable_layer_idx = batch.num_vars - 1
        logger.debug(
            f"free_variable_layer_idx: {free_variable_layer_idx.size()}"
        )
        free_var_emb = torch.gather(
            free_var_emb,
            1,
            free_variable_layer_idx.view(-1, 1, 1).expand(
                -1, 1, x_stacks.size(2)
            ),
        )

        logger.debug(f"free_var_emb: {free_var_emb.size()}")

        return free_var_emb

    def eval_all_entity_scores(self, batch: Batch):
        """
        Get the scores of all the entities.
        The score is defined in cosine similarity
        This function is intended for evaluation
        """
        femb = self.forward(batch).view(
            -1, 1, self.embedding_dim
        )  # [batch_size, embedding_dim]
        batch_size = 100
        begin = 0
        all_scores = []
        while begin < self.nbp.num_entities:
            end = min(begin + batch_size, self.nbp.num_entities)
            entity_ids = torch.arange(begin, end, device=femb.device)
            entity_embs = self.nbp.get_ent_emb(entity_ids).view(
                1, -1, self.embedding_dim
            )
            logger.debug(f"entity_embs: {entity_embs.size()}")
            logger.debug(f"femb: {femb.size()}")
            scores = F.cosine_similarity(femb, entity_embs, dim=2)
            all_scores.append(scores)
            begin = end

        scores = torch.cat(all_scores, dim=1)
        return scores

    def train_loss_nce(
        self,
        batch: Batch,
        answers: List[List[int]],
    ):
        """
        Compute the training loss.
        """
        femb = self.forward(batch)

        # get the one positive answer from all input answers
        pos_ans_tensor = (
            torch.tensor(
                [choice(answer_list) for answer_list in answers],
                device=femb.device,
            )
            .long()
            .view(-1)
        )

        # we sample the negative answers from the whole entity set
        neg_ans_tensor = torch.randint(
            0,
            self.nbp.num_entities,
            (1, self.negative_sample_size),
            device=femb.device,
        )

        # get the embeddings of the positive answers
        pos_emb = self.nbp.get_ent_emb(
            pos_ans_tensor
        )  # [batch_size, embedding_dim]
        neg_emb = self.nbp.get_ent_emb(
            neg_ans_tensor
        )  # [1, negative_sample_size, embedding_dim]

        # calculate the cosine similarity
        logging.debug(f"femb: {femb.size()}")
        logging.debug(f"pos_emb: {pos_emb.size()}")
        pos_scores = F.cosine_similarity(femb.squeeze(), pos_emb, dim=-1).view(
            -1, 1
        )  # [batch_size, 1]
        logging.debug(f"pos_scores: {pos_scores.size()}")

        logging.debug(f"neg_emb: {neg_emb.size()}")
        neg_scores = F.cosine_similarity(femb, neg_emb, dim=-1)

        all_scores = torch.cat([pos_scores, neg_scores], dim=1)

        logging.debug(f"neg_scores: {neg_scores.size()}")
        # calculate the loss
        loss = -pos_scores / self.T + torch.logsumexp(
            all_scores / self.T, dim=1
        )

        return loss.mean()

    def train_loss_softmax(self, batch: Batch, answers: List[List[int]]):
        """
        Compute the training loss.
        """
        fv_logits = self.eval_all_entity_scores(batch)

        boolean_target = torch.zeros_like(fv_logits)
        for i, t in enumerate(answers):
            boolean_target[i, t] = 1
        pred_pos = fv_logits * boolean_target
        max_n = torch.max(pred_pos, dim=-1)[0].unsqueeze(-1)
        loss = -F.log_softmax(fv_logits - max_n, dim=-1)[boolean_target.bool()]
        loss = loss.mean()
        return loss

    def train_loss(self, batch, answers):
        """
        Compute the training loss.
        """
        if self.loss == "nce":
            return self.train_loss_nce(batch, answers)
        elif self.loss == "softmax":
            return self.train_loss_softmax(batch, answers)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
