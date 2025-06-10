import json
import logging
import os
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Batch, Data
import tqdm


from knovex.language.efo_lang import EFOQuery, parse_lstr_to_lformula

logger = logging.getLogger(__name__)


def get_inverse_relation_id(relation: torch.Tensor):
    """
    the inverse relation is defined by its ID
    2i and 2i+1 are a pair of relations that are inverse to each other

    Args:
        relation (tensor): the relation ID

    using bit operation on pytorch tensors
    """

    return torch.bitwise_xor(
        relation, torch.Tensor([1]).bool().to(relation.device)
    )


def add_inverse_edge(graph: Data):
    """
    Add the inverse edge to the graph
    """

    edge_index = torch.cat(
        [
            graph.edge_index,
            torch.stack([graph.edge_index[1], graph.edge_index[0]], dim=0),
        ],
        dim=1,
    )
    edge_attr = torch.cat(
        [
            graph.edge_attr,
            torch.stack(
                [
                    get_inverse_relation_id(graph.edge_attr[:, 0]),
                    graph.edge_attr[:, 1],
                ],
                dim=1,
            ),
        ],
        dim=0,
    )

    graph.edge_index = edge_index
    graph.edge_attr = edge_attr

    return graph


class PyGAACollator:
    def __call__(
        self, batch_of_data: List[Tuple[List[Data], List[int], List[int]]]
    ) -> Tuple[Tuple[Batch], List[torch.Tensor], List[torch.Tensor]]:
        """
        This function is called by the DataLoader to collate a batch of data.
        batch_of_data is a list of tuples, where each tuple is in the format:
        (List[PyGData], List[int], List[int])

        Here we need to consider the DNF situation, where one query can have
        multiple conjunctive queries.

        The strategy here is that we externally ensure the number of conj
        queries to be consistent
        """
        # number of conjunctive queries this batch
        num_conj_query = len(batch_of_data[0][0])

        # get the list of PyGData objects
        pyg_data_list_by_conj_query = [[] for _ in range(num_conj_query)]
        easy_answer_list = []
        hard_answer_list = []
        for conj_query_list, easy_answer, hard_answer in batch_of_data:
            for i in range(num_conj_query):
                try:
                    pyg_data_list_by_conj_query[i].append(conj_query_list[i])
                except IndexError:
                    raise ValueError(
                        "Number of conj queries in the batch is inconsistent."
                    )
            easy_answer_list.append(easy_answer)
            hard_answer_list.append(hard_answer)

        batch_by_conj_query = [
            Batch.from_data_list(pyg_data_list, follow_batch=["num_vars"])
            for pyg_data_list in pyg_data_list_by_conj_query
        ]

        return batch_by_conj_query, easy_answer_list, hard_answer_list


class PyGAADataset:
    """
    Here are qaa graphs are converted into batches.
    """
    def __init__(self, qaa_file, add_inverse_edge=True):
        self.add_inverse_edge = add_inverse_edge
        self.qaa_file = qaa_file

        with open(qaa_file, "rt") as f:
            qaa = json.load(f)
        # load the qaa data and convert them into different queries
        self._data_query_dict: Dict[str, EFOQuery] = {}
        for lstr, _qaa in tqdm.tqdm(qaa.items(), desc="Loading qaa data"):
            query = EFOQuery.from_lstr(lstr)
            for q_inst_dict, easy_answer, hard_answer in _qaa:
                query.append_qaa_instance(
                    append_dict=q_inst_dict,
                    easy_answers=easy_answer,
                    hard_answers=hard_answer,
                )
            self._data_query_dict[parse_lstr_to_lformula(lstr).lstr()] = query

    def get_datalist_by_lstr(
        self, lstr_list=[]
    ) -> List[Tuple[List[Data], List[int], List[int]]]:
        """
        Each element in a datalist is in the following format.
        (
            List[PyGData]: for multiple query graphs in DNF
            List: for easy answers,
            List: for hard answers
        )
        """
        if len(lstr_list) == 0:
            lstr_list = list(self._data_query_dict.keys())
        else:
            logger.info(f"Using the following lstrs: {lstr_list}")

        datalist = []
        for lstr in tqdm.tqdm(lstr_list, desc="Converting qaa data to PyG"):
            cache_file = self.qaa_file.replace(".json", f"_{lstr}.pt")
            # if there is a cache file.
            if os.path.exists(cache_file):
                logger.info(f"Loading cache file {cache_file}")
                datalist.extend(torch.load(cache_file))
                continue

            # if there is no cache file, we need to convert the data
            query = self._data_query_dict[lstr]
            pyg_graph_list = (
                query.get_pyg_graph_list_by_sub_conjunctive_queries()
            )
            easy_answer_list = query.easy_answer_list
            hard_answer_list = query.hard_answer_list
            ans_key = '_'.join(query.free_variable_dict.keys())
            lstr_datalist = []
            for i, (easy_answer, hard_answer) in enumerate(
                zip(easy_answer_list, hard_answer_list)
            ):
                conj_pyg_list = [
                    add_inverse_edge(g[i]) if self.add_inverse_edge else g[i]
                    for g in pyg_graph_list
                ]

                _easy_answer = torch.tensor(easy_answer[ans_key], dtype=torch.long)
                if ans_key in hard_answer:
                    _hard_answer = torch.tensor(hard_answer[ans_key], dtype=torch.long)
                else:
                    _hard_answer = []

                lstr_datalist.append((conj_pyg_list, _easy_answer, _hard_answer))

            torch.save(lstr_datalist, cache_file)
            logger.info(f"Saving cache file {cache_file}")

            datalist.extend(lstr_datalist)
        return datalist
