import logging
from typing import Optional, List
from random import choice

from torch_geometric.nn import MessagePassing
from knovex.structure.kg.graph import KnowledgeGraph
from torch_geometric.data import Batch
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy


from knovex.structure.kg_embedding.abstract_kge import (
    KnowledgeGraphEmbedding as KGE,
)
from knovex.reasoner.abstract_reasoner import Reasoner
from knovex.layers.mlp import MLP

logger = logging.getLogger(__name__)

# Fuzzy Inference with Turth values (FIT)
# Make sure there exists relation matrices that are completed kg.


class FIT(nn.Module, Reasoner):
    def __init__(
        self,
        device,
        dtype=torch.float32,
        kg = None,
        loss="softmax",
        **kwargs,
    ):
        super(FIT, self).__init__()
        self.nbp: Optional[KGE] = None
        self.kg: Optional[KnowledgeGraph] = kg
        self.relation_matrix_list: List = []
        self.loss = loss
        self.c_norm, self.e_norm = "product", "Godel"
        self.max_total = 10

        self.device = device
        self.dtype = dtype
        if self.nbp:
            pass
        else:
            self.n_relation, self.n_entity = -1, -1

    def set_nbp(self, kge: KGE):
        pass

    def load_relation_matrix_list(self, ckpt_path):
        self.relation_matrix_list = torch.load(ckpt_path)
        self.n_relation, self.n_entity = len(self.relation_matrix_list), self.relation_matrix_list[0].shape[0]
        for i in range(len(self.relation_matrix_list)):
                self.relation_matrix_list[i] = self.relation_matrix_list[i].to(self.dtype).to(self.device)
        

    def construct_relation_matrix_list_with_nbp():
        assert self.kg != None
        pass


    def forward(self, batch: Batch):
        """
        Input pyg graph and output the fuzzy vectors of the free variables.
        """
        if len(self.relation_matrix_list) == self.n_relation:
            fuzzy_vec_fs = self.foward_with_relation_matrix_list(batch)
        elif self.nbp:
            fuzzy_vec_fs = self.foward_with_nbp(batch)
        else:
            print('Wrong setting for FIT.')
            assert True

        return fuzzy_vec_fs

    def foward_with_relation_matrix_list(self, batch: Batch):
        ans = solve_EFOX(batch, self.relation_matrix_list, self.c_norm, self.e_norm, 0, self.device, 0, self.max_total)

        return ans

    def foward_with_nbp(self, batch: Batch):
        pass

    def eval_all_entity_scores(self, batch: Batch):
        """
        Get the scores of all the entities.
        The score is inferred truth values.
        """
        scores = self.forward(batch)
        return scores

    def train_loss_nce(
        self,
        batch: Batch,
        answers: List[List[int]],
    ):
        """
        Compute the training loss.
        """
        pass

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

def kg_remove_node(kg: KnowledgeGraph, node: int):
    new_kg = deepcopy(kg)
    indices_reserved = torch.logical_and(new_kg.edge_index[0,:] != node, new_kg.edge_index[1,:] != node)
    new_kg.edge_index = new_kg.edge_index[:, indices_reserved]
    new_kg.edge_attr = new_kg.edge_attr[indices_reserved]

    return new_kg

@torch.no_grad()
def extend_ans(ans_node, sub_ans_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, relation_matrix,
               leaf_candidate, sub_ans, conj_tnorm, exist_tnorm):
    all_prob_matrix = construct_matrix_list(sub_ans_node, ans_node, sub_graph, neg_sub_graph, relation_matrix,
                                            conj_tnorm)
    if conj_tnorm == 'product':
        all_prob_matrix.mul_(sub_ans.unsqueeze(-1))
    elif conj_tnorm == 'Godel':
        all_prob_matrix = torch.minimum(all_prob_matrix, sub_ans.unsqueeze(-1))
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = (torch.amax(all_prob_matrix, dim=-2)).squeeze()  # prob*vec is 1*n  matrix
        del all_prob_matrix
    elif exist_tnorm == 'product':
        prob_vec = 1 - torch.prod(1 - all_prob_matrix, dim=-2)
        del all_prob_matrix
    else:
        raise NotImplementedError
    if conj_tnorm == 'product':
        prob_vec = leaf_candidate * prob_vec
    elif conj_tnorm == 'Godel':
        prob_vec = torch.minimum(leaf_candidate, prob_vec)
    else:
        raise NotImplementedError
    torch.cuda.empty_cache()
    return prob_vec

def ht2relationQG(QG, ht):
    n_edges = len(QG.edge_index[0])
    indices = torch.all(QG.edge_index[:, :n_edges//2] == torch.tensor(ht).unsqueeze(-1), dim=0)
    index_ht = torch.where(indices)[0]
    edges_ht = QG.edge_attr[index_ht]
    relations = set(edges_ht[:,0].tolist())

    return relations


@torch.no_grad()
def construct_matrix_list(head_node, tail_node, sub_graph, neg_sub_graph, relation_matrix_list, conj_tnorm):
    node_pair, reverse_node_pair = (head_node, tail_node), (tail_node, head_node)
    h2t_relation, t2h_relation = ht2relationQG(sub_graph, node_pair), ht2relationQG(sub_graph, reverse_node_pair)
    h2t_negation, t2h_negation = ht2relationQG(neg_sub_graph, node_pair), ht2relationQG(neg_sub_graph, reverse_node_pair)
    transit_matrix_list = []
    for r in h2t_relation:
        transit_matrix_list.append(relation_matrix_list[r])
    for r in t2h_relation:
        transit_matrix_list.append(relation_matrix_list[r].transpose(-2, -1))
    for r in h2t_negation:
        transit_matrix_list.append(1 - relation_matrix_list[r].to_dense())
    for r in t2h_negation:
        transit_matrix_list.append(1 - relation_matrix_list[r].transpose(-2, -1).to_dense())
    if conj_tnorm == 'product':
        all_prob_matrix = transit_matrix_list.pop(0)
        for i in range(len(transit_matrix_list)):
            if all_prob_matrix.is_sparse and not transit_matrix_list[i].is_sparse:
                all_prob_matrix = all_prob_matrix.to_dense()
                all_prob_matrix.mul_(transit_matrix_list[i])
            else:
                all_prob_matrix = all_prob_matrix.multiply(transit_matrix_list[i])
    elif conj_tnorm == 'Godel':
        all_prob_matrix = transit_matrix_list[0].to_dense() \
            if transit_matrix_list[0].is_sparse else transit_matrix_list[0]
        for i in range(1, len(transit_matrix_list)):
            if transit_matrix_list[i].is_sparse:
                all_prob_matrix = torch.minimum(all_prob_matrix, transit_matrix_list[i].to_dense())
            else:
                all_prob_matrix = torch.minimum(all_prob_matrix, transit_matrix_list[i])
    else:
        raise NotImplementedError
    if all_prob_matrix.is_sparse:  # n*n sparse matrix or dense matrix (when only one negation edges)
        return all_prob_matrix.to_dense()
    else:
        return all_prob_matrix



@torch.no_grad()
def existential_update(leaf_node, adjacency_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                       r_matrix_list, leaf_candidates, adj_candidates, conj_tnorm, exist_tnorm) -> dict:
    all_prob_matrix = construct_matrix_list(leaf_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list,
                                            conj_tnorm)

    if conj_tnorm == 'product':
        all_prob_matrix.mul_(leaf_candidates.unsqueeze(-1))
        all_prob_matrix.mul_(adj_candidates.unsqueeze(-2))
    elif conj_tnorm == 'Godel':
        all_prob_matrix = torch.minimum(all_prob_matrix, leaf_candidates.unsqueeze(-1))
        all_prob_matrix = torch.minimum(all_prob_matrix, adj_candidates.unsqueeze(-2))
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = torch.amax(all_prob_matrix, dim=-2).squeeze()
    else:
        raise NotImplementedError
    del all_prob_matrix
    torch.cuda.empty_cache()
    return prob_vec


@torch.no_grad()
def solve_EFOX(conj_formula, relation_matrix, conjunctive_tnorm, existential_tnorm, index, device,
               max_enumeration, max_enumeration_total):
    torch.cuda.empty_cache()
    with torch.no_grad():
        n_entity = relation_matrix[0].shape[0]
        all_candidates = {}
        for id_n, term_tensor in enumerate(conj_formula[0][0].x):
            if term_tensor[1] == 0:
                all_candidates[id_n] = torch.zeros(n_entity).to(device)
                all_candidates[id_n][term_tensor[0]] = 1
            else:
                all_candidates[id_n] = torch.ones(n_entity).to(device)
        indicate_pos = conj_formula[0][0].edge_attr[:, 1] == 0
        sub_graph_pos, sub_graph_neg = deepcopy(conj_formula[0][0]), deepcopy(conj_formula[0][0])
        sub_graph_pos.edge_attr = sub_graph_pos.edge_attr[indicate_pos]
        sub_graph_pos.edge_index = sub_graph_pos.edge_index[:, indicate_pos]

        sub_graph_neg.edge_attr = sub_graph_neg.edge_attr[~indicate_pos]
        sub_graph_neg.edge_index = sub_graph_neg.edge_index[:, ~indicate_pos]
        sub_graph_edge, sub_graph_negation_edge = [], []

        free_variable_list = [i for i, indicate_n in enumerate(conj_formula[0][0].x[:,1]) if indicate_n.item() == 2]
        free_variable_list.sort()
        sub_ans_dict = solve_conjunctive_all(sub_graph_pos, sub_graph_neg, relation_matrix,
                                    all_candidates, conjunctive_tnorm, existential_tnorm, free_variable_list, device,
                                    max_enumeration, max_enumeration_total, all_candidates)
        ans_emb_list = [sub_ans_dict[term_name] for term_name in free_variable_list]
        return torch.stack(ans_emb_list, dim=0)


def find_leaf_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable_list):
    """
    Find a leaf node with least possible candidate. The now-asking variable is first.
    """
    graphs = [sub_graph, neg_sub_graph]
    node2adjacency = defaultdict(set)
    for graph in graphs:
        for i in range(len(graph.edge_index[0])):
            node2adjacency[graph.edge_index[0][i].item()].add(graph.edge_index[1][i].item())
    return_candidate = [None, None, 0]
    for node in now_candidate:
        adjacency_node_set = node2adjacency[node]
        if len(adjacency_node_set) == 1:
            if node in now_variable_list:
                return node, list(adjacency_node_set)[0], True
            candidate_num = torch.count_nonzero(now_candidate[node])
            if not return_candidate[0] or candidate_num < return_candidate[2]:
                return_candidate = [node, list(adjacency_node_set)[0], candidate_num]
    return return_candidate[0], return_candidate[1], False


def find_enumerate_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable_list):
    graphs = [sub_graph, neg_sub_graph]
    node2adjacency = defaultdict(set)
    for graph in graphs:
        for i in range(len(graph.edge_index[0])):
            node2adjacency[graph.edge_index[0][i].item()].add(graph.edge_index[1][i].item())
    return_candidate = [None, 100, 100000]
    for node in now_candidate:
        # if node in now_variable_list:
        #     continue
        adjacency_node_list = node2adjacency[node]
        adjacency_node_num = len(adjacency_node_list)
        candidate_num = torch.count_nonzero(now_candidate[node])
        if not return_candidate[0] or adjacency_node_num < len(return_candidate[1]) or \
                (adjacency_node_num == len(return_candidate[1]) and candidate_num < return_candidate[2]):
            return_candidate = node, adjacency_node_list, candidate_num
    return return_candidate[0], return_candidate[1]


@torch.no_grad()
def solve_conjunctive_all(positive_graph: KnowledgeGraph, negative_graph: KnowledgeGraph, relation_matrix,
                      now_candidate_set: dict, conjunctive_tnorm, existential_tnorm, now_variable_list, device,
                      max_enumeration, max_enumeration_total, all_candidate_set):
    n_entity = relation_matrix[0].shape[0]
    if not len(positive_graph.edge_index[0]) and not len(negative_graph.edge_index[0]):
        return all_candidate_set
    if len(now_candidate_set) == 1:
        return all_candidate_set
    now_leaf_node, adjacency_node, being_asked_variable = \
        find_leaf_node(positive_graph, negative_graph, now_candidate_set, now_variable_list)
    if now_leaf_node:  # If there exists leaf node in the query graph, always possible to shrink into a sub_problem.
        adjacency_node_list = [adjacency_node]
        if being_asked_variable:
            # next_variable = adjacency_node
            sub_pos_g, sub_neg_g = kg_remove_node(positive_graph, now_leaf_node), \
                                   kg_remove_node(negative_graph, now_leaf_node)
            copy_variable_list = deepcopy(now_variable_list)
            copy_variable_list.remove(now_leaf_node)
            sub_ans_dict = solve_conjunctive_all(sub_pos_g, sub_neg_g, relation_matrix, now_candidate_set,
                                                 conjunctive_tnorm, existential_tnorm, copy_variable_list, device,
                                                 max_enumeration, max_enumeration_total, all_candidate_set)
            final_ans = extend_ans(now_leaf_node, adjacency_node, positive_graph, negative_graph, relation_matrix,
                                   now_candidate_set[now_leaf_node], sub_ans_dict[adjacency_node], conjunctive_tnorm,
                                   existential_tnorm)
            all_candidate_set[now_leaf_node] = final_ans
            return all_candidate_set
        else:
            sub_candidate_set = cut_node_sub_problem(now_leaf_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable_list, device, max_enumeration, max_enumeration_total,
                                                     all_candidate_set)
            return sub_candidate_set
    else:
        to_enumerate_node, adjacency_node_list = find_enumerate_node(positive_graph, negative_graph, now_candidate_set,
                                                                     now_variable_list)
        easy_candidate = torch.count_nonzero(now_candidate_set[to_enumerate_node] == 1)
        enumeration_num = torch.count_nonzero(now_candidate_set[to_enumerate_node])
        max_enumeration_here = min(max_enumeration + easy_candidate, max_enumeration_total)
        if torch.count_nonzero(now_candidate_set[to_enumerate_node]) > 100:
            sub_candidate_set = cut_node_sub_problem(to_enumerate_node, adjacency_node_list, positive_graph, negative_graph,
                                 relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                 now_variable_list, device, max_enumeration, max_enumeration_total, all_candidate_set)
            return sub_candidate_set
        if max_enumeration is not None:
            to_enumerate_candidates = torch.argsort(now_candidate_set[to_enumerate_node],
                                                    descending=True)[:min(max_enumeration_here, enumeration_num)]
        else:
            to_enumerate_candidates = now_candidate_set[to_enumerate_node].nonzero()
        this_node_candidates = deepcopy(now_candidate_set[to_enumerate_node])
        all_enumerate_ans = torch.zeros((to_enumerate_candidates.shape[0], n_entity)).to(device)
        if to_enumerate_candidates.shape[0] == 0:
            return {variable: torch.zeros(n_entity).to(device) for variable in all_candidate_set}
        for i, enumerate_candidate in enumerate(to_enumerate_candidates):
            single_candidate = torch.zeros_like(now_candidate_set[to_enumerate_node]).to(device)
            candidate_truth_value = this_node_candidates[enumerate_candidate]
            single_candidate[enumerate_candidate] = 1
            now_candidate_set[to_enumerate_node] = single_candidate
            answer_dict = cut_node_sub_problem(to_enumerate_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable_list, device, max_enumeration, max_enumeration_total,
                                               all_candidate_set)
            for free_variable in now_variable_list:
                answer = answer_dict[free_variable]
                if conjunctive_tnorm == 'product':
                    enumerate_ans = candidate_truth_value * answer
                elif conjunctive_tnorm == 'Godel':
                    enumerate_ans = torch.minimum(candidate_truth_value, answer)
                else:
                    raise NotImplementedError
                all_enumerate_ans[i] = enumerate_ans
                if existential_tnorm == 'Godel':
                    final_ans = torch.amax(all_enumerate_ans, dim=-2)
                else:
                    raise NotImplementedError
                all_candidate_set[free_variable] = final_ans
        return all_candidate_set


@torch.no_grad()
def cut_node_sub_problem(to_cut_node, adjacency_node_list, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                         r_matrix_list, now_candidate_set, conj_tnorm, exist_tnorm, now_variable, device,
                         max_enumeration, max_enumeration_total, all_candidate_set):
    new_candidate_set = deepcopy(now_candidate_set)
    for adjacency_node in adjacency_node_list:
        adj_candidate_vec = existential_update(to_cut_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list,
                                               new_candidate_set[to_cut_node], new_candidate_set[adjacency_node],
                                               conj_tnorm, exist_tnorm)
        new_candidate_set[adjacency_node] = adj_candidate_vec
        all_candidate_set[adjacency_node] = adj_candidate_vec
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    new_candidate_set.pop(to_cut_node)
    sub_answer = solve_conjunctive_all(new_sub_graph, new_sub_neg_graph, r_matrix_list, new_candidate_set, conj_tnorm,
                                       exist_tnorm, now_variable, device, max_enumeration, max_enumeration_total,
                                       all_candidate_set)
    return sub_answer


def compute_single_evaluation(fof, batch_ans_tensor, n_entity, eval_device):
    argsort = torch.argsort(batch_ans_tensor, dim=-1, descending=True)
    ranking = argsort.clone().to(torch.float).to(eval_device)
    ranking = ranking.scatter_(2, argsort, torch.arange(n_entity).to(torch.float).
                               repeat(argsort.shape[0], argsort.shape[1], 1).to(eval_device))
    two_marginal_logs = defaultdict(float)
    one_marginal_logs = defaultdict(float)
    no_marginal_logs = defaultdict(float)
    f_str_list = [f'f{i + 1}' for i in range(len(fof.free_term_dict))]
    f_str = '_'.join(f_str_list)
    if len(fof.free_term_dict) == 1:
        with torch.no_grad():
            ranking.squeeze_()
            for i in range(batch_ans_tensor.shape[0]):
                # ranking = ranking.scatter_(0, argsort, torch.arange(n_entity).to(torch.float))
                easy_ans = [instance[0] for instance in fof.easy_answer_list[i][f_str]]
                hard_ans = [instance[0] for instance in fof.hard_answer_list[i][f_str]]
                mrr, h1, h3, h10 = ranking2metrics(ranking[i], easy_ans, hard_ans, eval_device)
                two_marginal_logs['MRR'] += mrr
                two_marginal_logs['HITS1'] += h1
                two_marginal_logs['HITS3'] += h3
                two_marginal_logs['HITS10'] += h10
            two_marginal_logs['num_queries'] += batch_ans_tensor.shape[0]
            return two_marginal_logs, one_marginal_logs, no_marginal_logs
    else:
        with torch.no_grad():
            three_marginal_logs, two_marginal_logs, one_marginal_logs, no_marginal_logs = evaluate_batch_joint(
                ranking, fof.easy_answer_list, fof.hard_answer_list, eval_device, f_str)
            return three_marginal_logs, two_marginal_logs, one_marginal_logs, no_marginal_logs
