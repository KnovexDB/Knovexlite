# Add FIT reasoner implementation referencing the FIT repository
import copy
from typing import Dict, List

import torch
from torch_geometric.data import Batch, Data

from knovex.reasoner.abstract_reasoner import Reasoner
from knovex.structure.kg.graph import KnowledgeGraph
from knovex.structure.kg.index import KGIndex
from knovex.structure.kg_embedding.abstract_kge import KnowledgeGraphEmbedding as KGE


class FITReasoner(Reasoner):
    """Symbolic FIT reasoner."""

    def __init__(
        self,
        conj_tnorm: str = "product",
        exist_tnorm: str = "Godel",
        max_enumeration: int = 10,
    ) -> None:
        self.conj_tnorm = conj_tnorm
        self.exist_tnorm = exist_tnorm
        self.max_enumeration = max_enumeration

        self.nbp: KGE | None = None
        self._relation_matrix_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _relation_matrix(self, rel_id: int) -> torch.Tensor:
        assert self.nbp is not None
        if rel_id not in self._relation_matrix_cache:
            n = self.nbp.num_entities
            device = self.nbp.device
            h = torch.arange(n, device=device).view(n, 1).repeat(1, n).view(-1)
            t = torch.arange(n, device=device).repeat(n)
            r = torch.full_like(h, rel_id)
            score = self.nbp.constraint_score(h, r, tail_id=t)
            self._relation_matrix_cache[rel_id] = score.view(n, n)
        return self._relation_matrix_cache[rel_id]

    def _construct_matrix_list(
        self,
        h: int,
        t: int,
        pos_graph: KnowledgeGraph,
        neg_graph: KnowledgeGraph,
    ) -> List[torch.Tensor]:
        matrices = []
        for r in pos_graph.ht2r[(h, t)]:
            matrices.append(self._relation_matrix(r))
        for r in pos_graph.ht2r[(t, h)]:
            matrices.append(self._relation_matrix(r).t())
        for r in neg_graph.ht2r[(h, t)]:
            matrices.append(1 - self._relation_matrix(r))
        for r in neg_graph.ht2r[(t, h)]:
            matrices.append(1 - self._relation_matrix(r).t())
        return matrices

    def _agg_matrices(self, matrices: List[torch.Tensor]) -> torch.Tensor:
        assert len(matrices) > 0
        out = matrices[0]
        if self.conj_tnorm == "product":
            for m in matrices[1:]:
                out = out * m
        elif self.conj_tnorm == "Godel":
            for m in matrices[1:]:
                out = torch.minimum(out, m)
        else:
            raise NotImplementedError
        return out

    # ------------------------------------------------------------------
    # Core algorithm adapted from https://github.com/HKUST-KnowComp/FIT
    # ------------------------------------------------------------------
    def _extend_ans(
        self,
        ans_node: int,
        sub_node: int,
        pos_graph: KnowledgeGraph,
        neg_graph: KnowledgeGraph,
        leaf_cand: torch.Tensor,
        sub_ans: torch.Tensor,
    ) -> torch.Tensor:
        matrices = self._construct_matrix_list(sub_node, ans_node, pos_graph, neg_graph)
        mat = self._agg_matrices(matrices)
        if self.conj_tnorm == "product":
            mat = mat * sub_ans.unsqueeze(-1)
        else:
            mat = torch.minimum(mat, sub_ans.unsqueeze(-1))
        if self.exist_tnorm == "Godel":
            prob = mat.amax(dim=-2)
        else:
            prob = 1 - torch.prod(1 - mat, dim=-2)
        if self.conj_tnorm == "product":
            prob = leaf_cand * prob
        else:
            prob = torch.minimum(leaf_cand, prob)
        return prob

    def _existential_update(
        self,
        leaf: int,
        adj: int,
        pos_graph: KnowledgeGraph,
        neg_graph: KnowledgeGraph,
        leaf_cand: torch.Tensor,
        adj_cand: torch.Tensor,
    ) -> torch.Tensor:
        matrices = self._construct_matrix_list(leaf, adj, pos_graph, neg_graph)
        mat = self._agg_matrices(matrices)
        if self.conj_tnorm == "product":
            mat = mat * leaf_cand.unsqueeze(-1)
            mat = mat * adj_cand.unsqueeze(-2)
        else:
            mat = torch.minimum(mat, leaf_cand.unsqueeze(-1))
            mat = torch.minimum(mat, adj_cand.unsqueeze(-2))
        if self.exist_tnorm == "Godel":
            prob = mat.amax(dim=-2)
        else:
            prob = 1 - torch.prod(1 - mat, dim=-2)
        return prob

    def _kg_remove_node(self, kg: KnowledgeGraph, node: int) -> KnowledgeGraph:
        triples = [tr for tr in kg.triples if tr[0] != node and tr[2] != node]
        return KnowledgeGraph(triples, kg.kgindex)

    def _find_leaf_node(
        self,
        pos_graph: KnowledgeGraph,
        neg_graph: KnowledgeGraph,
        candidate: Dict[int, torch.Tensor],
        ask_var: int,
    ) -> tuple[int | None, int | None, bool]:
        ret = (None, None, False)
        best_num = float("inf")
        for node in candidate:
            adj = set().union(
                pos_graph.h2t[node],
                pos_graph.t2h[node],
                neg_graph.h2t[node],
                neg_graph.t2h[node],
            )
            if len(adj) == 1:
                num = candidate[node].nonzero().size(0)
                if num < best_num:
                    ret = (node, next(iter(adj)), node == ask_var)
                    best_num = num
        return ret

    def _find_enum_node(
        self,
        pos_graph: KnowledgeGraph,
        neg_graph: KnowledgeGraph,
        candidate: Dict[int, torch.Tensor],
        ask_var: int,
    ) -> tuple[int | None, List[int]]:
        ret = (None, [])
        best_adj = 100
        best_cand = float("inf")
        for node in candidate:
            if node == ask_var:
                continue
            adj = list(
                set().union(
                    pos_graph.h2t[node],
                    pos_graph.t2h[node],
                    neg_graph.h2t[node],
                    neg_graph.t2h[node],
                )
            )
            num_adj = len(adj)
            num_cand = candidate[node].nonzero().size(0)
            if num_adj < best_adj or (num_adj == best_adj and num_cand < best_cand):
                ret = (node, adj)
                best_adj = num_adj
                best_cand = num_cand
        return ret

    def _cut_node_subproblem(
        self,
        node: int,
        adj_list: List[int],
        pos_graph: KnowledgeGraph,
        neg_graph: KnowledgeGraph,
        candidate: Dict[int, torch.Tensor],
        ask_var: int,
    ) -> torch.Tensor:
        new_cand = copy.deepcopy(candidate)
        for adj in adj_list:
            updated = self._existential_update(
                node,
                adj,
                pos_graph,
                neg_graph,
                new_cand[node],
                new_cand[adj],
            )
            new_cand[adj] = updated
        pos_graph = self._kg_remove_node(pos_graph, node)
        neg_graph = self._kg_remove_node(neg_graph, node)
        new_cand.pop(node)
        return self._solve_conjunctive(pos_graph, neg_graph, new_cand, ask_var)

    def _solve_conjunctive(
        self,
        pos_graph: KnowledgeGraph,
        neg_graph: KnowledgeGraph,
        candidate: Dict[int, torch.Tensor],
        ask_var: int,
    ) -> torch.Tensor:
        n = self.nbp.num_entities
        if not pos_graph.triples and not neg_graph.triples:
            return candidate[ask_var]
        if len(candidate) == 1:
            return candidate[ask_var]
        leaf, adj, is_ask = self._find_leaf_node(pos_graph, neg_graph, candidate, ask_var)
        if leaf is not None:
            if is_ask:
                next_var = adj
                sub_pos = self._kg_remove_node(pos_graph, leaf)
                sub_neg = self._kg_remove_node(neg_graph, leaf)
                sub_ans = self._solve_conjunctive(sub_pos, sub_neg, candidate, next_var)
                return self._extend_ans(
                    leaf,
                    adj,
                    pos_graph,
                    neg_graph,
                    candidate[leaf],
                    sub_ans,
                )
            return self._cut_node_subproblem(
                leaf,
                [adj],
                pos_graph,
                neg_graph,
                candidate,
                ask_var,
            )
        enum_node, adj_list = self._find_enum_node(pos_graph, neg_graph, candidate, ask_var)
        if enum_node is None:
            return torch.zeros(n, device=self.nbp.device)
        candidates = candidate[enum_node].nonzero(as_tuple=False).view(-1)
        if self.max_enumeration:
            easy = (candidate[enum_node] == 1).nonzero(as_tuple=False).size(0)
            num = candidates.size(0)
            k = min(self.max_enumeration + easy, num)
            topk = torch.topk(candidate[enum_node], k).indices
            candidates = topk
        all_ans = []
        for c in candidates:
            single = torch.zeros_like(candidate[enum_node])
            single[c] = 1
            new_cand = copy.deepcopy(candidate)
            cand_val = candidate[enum_node][c]
            new_cand[enum_node] = single
            ans = self._cut_node_subproblem(enum_node, adj_list, pos_graph, neg_graph, new_cand, ask_var)
            if self.conj_tnorm == "product":
                all_ans.append(cand_val * ans)
            else:
                all_ans.append(torch.minimum(cand_val, ans))
        stack = torch.stack(all_ans)
        if self.exist_tnorm == "Godel":
            return stack.amax(dim=0)
        return 1 - torch.prod(1 - stack, dim=0)

    # ------------------------------------------------------------------
    def set_nbp(self, kge: KGE) -> None:  # type: ignore[override]
        self.nbp = kge
        self._relation_matrix_cache.clear()

    # ------------------------------------------------------------------
    def train_loss(self, batch: Batch, target: torch.Tensor):  # type: ignore[override]
        raise NotImplementedError("FIT reasoner is symbolic and not trainable")

    def _data_to_graphs(self, data: Data) -> tuple[KnowledgeGraph, KnowledgeGraph, Dict[int, torch.Tensor], int]:
        assert self.nbp is not None
        pos_edges: List[tuple[int, int, int]] = []
        neg_edges: List[tuple[int, int, int]] = []
        for i in range(data.edge_index.size(1)):
            h = int(data.edge_index[0, i])
            t = int(data.edge_index[1, i])
            r = int(data.edge_attr[i, 0])
            nflag = int(data.edge_attr[i, 1])
            if nflag:
                neg_edges.append((h, r, t))
            else:
                pos_edges.append((h, r, t))
        kgindex = KGIndex()
        for nid in range(data.num_nodes):
            kgindex.register_entity(str(nid), eid=nid)
        for _, r, _ in pos_edges + neg_edges:
            kgindex.register_relation(str(r), rid=r)
        pos_graph = KnowledgeGraph(pos_edges, kgindex)
        neg_graph = KnowledgeGraph(neg_edges, kgindex)
        candidates: Dict[int, torch.Tensor] = {}
        for nid in range(data.num_nodes):
            label = int(data.x[nid, 1])
            if label == 0:
                cand = torch.zeros(self.nbp.num_entities, device=self.nbp.device)
                cand[int(data.x[nid, 0])] = 1
            else:
                cand = torch.ones(self.nbp.num_entities, device=self.nbp.device)
            candidates[nid] = cand
        ask_var = int(torch.nonzero(data.x[:, 1] == 2).squeeze())
        return pos_graph, neg_graph, candidates, ask_var

    def eval_all_entity_scores(self, batch: Batch):  # type: ignore[override]
        assert self.nbp is not None
        results = []
        for data in batch.to_data_list():
            pos_g, neg_g, cand, ask = self._data_to_graphs(data)
            ans = self._solve_conjunctive(pos_g, neg_g, cand, ask)
            results.append(ans)
        return torch.stack(results)
