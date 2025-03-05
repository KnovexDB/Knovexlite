import logging
from typing import Optional, List, Dict, Tuple

from torch_geometric.data import Batch
from torch_geometric.utils import index_to_mask
import torch
from torch import nn
from torch.nn import functional as F

from ngdb.structure.kg_embedding.abstract_kge import (
    KnowledgeGraphEmbedding as KGE,
)
from ngdb.reasoner.abstract_reasoner import Reasoner
from ngdb.utils.data import VariadicMatrix, aggregation

logger = logging.getLogger(__name__)


class CQDBeam(Reasoner, nn.Module):
    """
    Similar to other reasoners, this reasoner only works for PyG graphs.

    For CQD reasoner, the core algorithm is
    1. first, find the index of the free variables
    2. then, based on the edge_index in batch, construct the list of nodes
    that are connected to the free variables in BFS. Meanwhile, save the
    predicates and negation in edge_attr in the same order.
    3. The backward search list is completed untill it reaches all constant
    nodes.
    4. then, run the forward search algorithm to get the scores of all entities
    """

    __cache_path = "outputs/cache/cqd_beam"

    def __init__(
        self,
        beam_size: int,
        nbp: Optional[KGE] = None,
        **kwargs,
    ):
        super(CQDBeam, self).__init__()
        self.beam_size = beam_size

        self.nbp: KGE = nbp
        self._search_cache: List[Dict] = []

    def cache_results_of_leaf_existential(self):
        """
        Cache the results of leaf existential nodes.
        """
        # create the folder if not exists
        import os

        if not os.path.exists(self.__cache_path):
            os.makedirs(self.__cache_path)

        # if the cache file is not empty then save the cache

    def load_results_of_leaf_existential(self):
        """
        Load the results of leaf existential nodes.
        """
        pass

    def set_nbp(self, nbp: KGE):
        self.nbp = nbp

    def train_loss(self, batch: Batch, answers: List[List[int]]):
        """
        Compute the training loss.
        """
        # assert self.adapt_flag, "CQD is not adapted so no need to train"

        scores = self.eval_all_entity_scores(batch)

        target = torch.zeros_like(scores)
        for i, answer in enumerate(answers):
            target[i, answer] = 1

        return F.binary_cross_entropy(scores, target)

    def eval_all_entity_scores(self, batch):
        return self._eval_all_entity_scores_variadic_matrix(batch)

    def _eval_all_entity_scores_variadic_matrix(self, batch: Batch):

        # this should be set false because the variadic matrix version
        # does not match the score calculation in the pytorch native version.
        VM_FLAG = False
        ################### Conditions ###################
        # scalars
        device = batch.x.device
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        num_terms = batch.x.size(0)

        logger.debug("start a problem")
        logger.debug(f"num_terms: {num_terms}")
        logger.debug(f"batch.x: {batch.x}")
        logger.debug(f"edge_index: {edge_index}")
        logger.debug(f"edge_attr: {edge_attr}")

        # constant terms information
        constant_term_id = (batch.x[:, 1] == 0).nonzero().squeeze()
        constant_mask = index_to_mask(constant_term_id, size=num_terms)

        # algorithm states
        visited_mask = constant_mask.new_zeros(num_terms)

        ################### Functions ###################
        def recursive_beam_search(
            target_term_id: torch.Tensor,
            beam_size: int = -1,
        ) -> Tuple[VariadicMatrix, VariadicMatrix]:
            """
            Conduct the beam search recursively.

            1. We guarantee that target_term_id is for variables.
            2. we find the source_terms of two folds
                1. source_vars, which feed into next level of recursion
                2. source_consts, which we read the results directly
            Args:
                - target_term_id: indicates the target terms we want to search,
                    id is given in the batch.
                - beam_size, the beam size for the search, only valid in the
                    outmost call. Default is -1, which will be manually set to
                    self.beam_size. Recursive call inside this function will
                    be used as -1
            Returns: a pair of tensors (assignment, score)
                - assignment [#term_id, beam_size]
                - scores [#term_id, beam_size]
            """

            # update visited_mask
            visited_mask[target_term_id] = True

            target_term_mask = index_to_mask(target_term_id, size=num_terms)
            # also, filter s2t edges by unvisited target.
            active_edge_mask = (
                target_term_mask[edge_index[1]] & ~visited_mask[edge_index[0]]
            )

            active_edge_index = edge_index[:, active_edge_mask]
            active_edge_attr = edge_attr[active_edge_mask]
            num_active_edges = active_edge_mask.sum().item()

            # If no active edges, then the target nodes are existential leaves.
            if num_active_edges == 0:
                all_entity = torch.arange(
                    self.nbp.num_entities,
                    dtype=torch.long,
                    device=device,
                ).repeat_interleave(target_term_id.size(0))
                row_index = target_term_id.repeat(self.nbp.num_entities)

                assg_entity = VariadicMatrix(
                    data=all_entity, row_index=row_index
                )
                assg_score = VariadicMatrix(
                    data=torch.ones_like(all_entity, dtype=torch.float),
                    row_index=row_index,
                )

                return assg_entity, assg_score

            #### working on source part ####
            # get the source term id, which is the head of the edge
            # into_edge_mask
            edge_src_term_id = active_edge_index[0]

            term_src_mask = index_to_mask(edge_src_term_id, size=num_terms)

            # split by the source term is constant or variable
            term_src_const_mask = term_src_mask & constant_mask

            # prepare the source assignment and scores on constants
            if term_src_const_mask.sum() > 0:
                src_const_id = term_src_const_mask.nonzero().squeeze()
                term_src_assgmt_vm = VariadicMatrix(
                    data=batch.x[src_const_id, 0],
                    row_index=src_const_id,
                )
                term_src_scores_vm = VariadicMatrix(
                    data=torch.ones(
                        len(src_const_id), dtype=torch.float, device=device
                    ),
                    row_index=src_const_id,
                )
            else:
                term_src_scores_vm, term_src_assgmt_vm = None, None

            # if there are any source variables, we need to search further
            term_src_var_mask = term_src_mask & ~constant_mask
            if term_src_var_mask.sum() > 0:
                # first, handle variables if any, we need to search further
                src_var_id = term_src_var_mask.nonzero().squeeze()
                src_var_assgmt, src_var_scores = recursive_beam_search(
                    src_var_id, beam_size=self.beam_size
                )

                # merge the results
                if term_src_assgmt_vm is None:
                    term_src_assgmt_vm = src_var_assgmt
                else:
                    term_src_assgmt_vm.append(src_var_assgmt)

                if term_src_scores_vm is None:
                    term_src_scores_vm = src_var_scores
                else:
                    term_src_scores_vm.append(src_var_scores)

            # now the source assignment and scores are ready in term-wise
            # for sanity, sort and unique the assgmt and scores.
            term_src_scores_vm.index_global_(term_src_assgmt_vm.sort_())
            term_src_scores_vm.index_global_(
                term_src_assgmt_vm.unique_consecutive_()
            )

            # next compute the constraint scores from source to target
            # prepare source assignment in edge-wise manner
            edge_src_assgmt_vm = term_src_assgmt_vm.index_on_rows(
                edge_src_term_id, reindex=True
            )
            edge_src_scores_vm = term_src_scores_vm.index_on_rows(
                edge_src_term_id, reindex=True
            )

            # prepare the relation and negations for the further computation
            relation_vm = VariadicMatrix(
                data=active_edge_attr[:, 0],
                row_index=torch.arange(len(active_edge_attr), device=device),
            )
            negation_vm = VariadicMatrix(
                data=active_edge_attr[:, 1],
                row_index=torch.arange(len(active_edge_attr), device=device),
            )

            # join them in cartisian product, because each edge only has
            # one relation and negation, we only needs to care about the
            # one side reult of cartisian product.
            relation_vm.row_expand_as_(edge_src_assgmt_vm)
            negation_vm.row_expand_as_(edge_src_assgmt_vm)

            constraint_scores = self.nbp.constraint_score(
                head_id=edge_src_assgmt_vm.data,
                rel_id=relation_vm.data,
                negation=negation_vm.data,
                tail_id=None,
            )

            if VM_FLAG:
                # this piece is skipped for debuging
                # if self.conj_aggr == "depreciated_min":
                #     constraint_scores = torch.minimum(
                #         constraint_scores.view(-1),
                #         edge_src_scores_vm.data.repeat_interleave(
                #             self.nbp.num_entities
                #         ),
                #     )
                # elif self.conj_aggr == "depreciated_mul":
                #     constraint_scores = constraint_scores.view(
                #         -1
                #     ) * edge_src_scores_vm.data.repeat_interleave(
                #         self.nbp.num_entities
                #     )
                # else:
                constraint_scores = constraint_scores.view(
                    -1
                ) + edge_src_scores_vm.data.repeat_interleave(
                    self.nbp.num_entities
                )

                constraint_scores = constraint_scores.view(-1)

                edge_tgt_assgmt_vm = VariadicMatrix(
                    data=torch.arange(
                        self.nbp.num_entities,
                        dtype=torch.long,
                        device=device,
                    ).repeat(edge_src_assgmt_vm.size),
                    row_index=edge_src_assgmt_vm.row_index.repeat_interleave(
                        self.nbp.num_entities
                    ),
                )

                edge_tgt_scores_vm, edge_tgt_assgmt_vm = aggregation(
                    constraint_scores,
                    edge_tgt_assgmt_vm,
                    reduce="max",
                )

                # AGGR2 from edge-wise to term-wise
                # to show this, we need to change the row_index into term-wise
                new_row_index = active_edge_index[
                    1, edge_tgt_assgmt_vm.row_index
                ]
                term_assgmt_vm = VariadicMatrix(
                    data=edge_tgt_assgmt_vm.data,
                    row_index=new_row_index,
                )
                term_scores_vm = VariadicMatrix(
                    data=edge_tgt_scores_vm.data,
                    row_index=new_row_index,
                )
                logger.debug(
                    f"term_assgmt_vm.data: {term_assgmt_vm.data.size()}"
                )

                # we need to aggregate constraints into terms.
                term_scores_vm.index_global_(term_assgmt_vm.sort_())

                term_scores_vm, term_assgmt_vm = aggregation(
                    term_scores_vm.data,
                    term_assgmt_vm,
                    reduce="sum",
                )
            else:
                # implement a non-vm version because the shape is good
                edge_src_scores = edge_src_scores_vm.data.view(-1, 1)
                edge_scores = constraint_scores + edge_src_scores

                # then we aggregate the edge scores with the remove of head
                unique_edge_index, inverse_index = (
                    edge_src_assgmt_vm.row_index.unique(return_inverse=True)
                )

                edge_tgt_scores = torch.scatter_reduce(
                    input=edge_scores.new_zeros(
                        size=(len(unique_edge_index), self.nbp.num_entities)
                    ),
                    dim=0,
                    index=inverse_index.unsqueeze(1).expand(
                        -1, self.nbp.num_entities
                    ),
                    src=edge_scores,
                    reduce="max",
                    include_self=False,
                )

                unique_term_index, inverse_index = active_edge_index[
                    1, unique_edge_index
                ].unique(return_inverse=True)

                term_tgt_scores = torch.scatter_reduce(
                    input=edge_tgt_scores.new_zeros(
                        size=(len(unique_term_index), self.nbp.num_entities)
                    ),
                    dim=0,
                    index=inverse_index.unsqueeze(1).expand(
                        -1, self.nbp.num_entities
                    ),
                    src=edge_tgt_scores,
                    reduce="sum",
                    include_self=False,
                )

                term_row_index = unique_term_index.repeat_interleave(
                    self.nbp.num_entities
                )
                term_assgmt_vm = VariadicMatrix(
                    data=torch.arange(
                        self.nbp.num_entities,
                        dtype=torch.long,
                        device=device,
                    ).repeat(len(unique_term_index)),
                    row_index=term_row_index,
                )

                term_scores_vm = VariadicMatrix(
                    data=term_tgt_scores.view(-1),
                    row_index=term_row_index,
                )

            if beam_size > 0:
                # let's discard the variadic matrix topk and embraces the torch

                if VM_FLAG:
                    term_assgmt_vm.index_global_(
                        term_scores_vm.sort_(descending=True)
                    )
                    topk_index = term_scores_vm.topk_index(k=beam_size)
                    term_assgmt_vm.index_global_(topk_index)
                    term_scores_vm.index_global_(topk_index)
                else:
                    unique_index = term_assgmt_vm.reindex_()
                    term_scores_vm.reindex_()

                    term_assgmt = term_assgmt_vm.to_dense()
                    term_scores = term_scores_vm.to_dense()

                    topk_index = torch.topk(
                        term_scores,
                        beam_size,
                        dim=1,
                        largest=True,
                        sorted=False,
                    )[1]

                    term_assgmt_topk = term_assgmt.gather(1, topk_index)
                    term_scores_topk = term_scores.gather(1, topk_index)

                    term_assgmt_vm = VariadicMatrix(
                        data=term_assgmt_topk.view(-1),
                        row_index=unique_index.repeat_interleave(beam_size),
                    )
                    term_scores_vm = VariadicMatrix(
                        data=term_scores_topk.view(-1),
                        row_index=unique_index.repeat_interleave(beam_size),
                    )

            logger.debug(f"term_assgmt_vm.data: {term_assgmt_vm.data}")
            logger.debug(f"term_scores_vm.data: {term_scores_vm.data}")

            return term_assgmt_vm, term_scores_vm

        ################### Main ###################
        # prepare the target term ids
        target_term_id = (batch.x[:, 1] == 2).nonzero().squeeze()
        # start the recursive beam search
        _, score_vm = recursive_beam_search(target_term_id)
        score_vm.reindex_()
        score = score_vm.to_dense(padding_value=torch.nan)
        if torch.isnan(score).any():
            logger.warning(
                "There are NaNs in the scores, please check the code."
            )
            logger.warning(
                f"NaNs are found in the scores position: {torch.isnan(score).nonzero()}"
            )

        return score
