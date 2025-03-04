from typing import Union, List, Tuple
from itertools import chain
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import scatter

logger = logging.getLogger(__name__)


def _iter_triple_from_tsv(triple_file, to_int, check_size):
    with open(triple_file, "rt") as f:
        for line in f.readlines():
            triple = line.strip().split()
            if check_size:
                assert len(triple) == check_size
            if to_int:
                triple = [int(t) for t in triple]
            yield triple


def iter_triple_from_tsv(
    triple_files, to_int: bool = True, check_size: int = 3
):
    if isinstance(triple_files, list):
        return chain(*[iter_triple_from_tsv(tfile) for tfile in triple_files])
    elif isinstance(triple_files, str):
        return _iter_triple_from_tsv(triple_files, to_int, check_size)
    else:
        raise NotImplementedError("invalid input of triple files")


def tensorize_batch_entities(
    entities: Union[List[int], List[List[int]], torch.Tensor], device
) -> torch.Tensor:
    """
    convert the entities into the tensor formulation
    in the shape of [batch_size, num_entities]
    we interprete three cases
    1. List[int] batch size = 1
    2. List[List[int]], each inner list is a sample
    3. torch.Tensor in shape [batch_size, num_entities]
    """
    if isinstance(entities, list):
        if isinstance(entities[0], int):
            # in this case, batch size = 1
            entity_tensor = torch.tensor(entities, device=device).reshape(
                -1, 1
            )
        elif isinstance(entities[0], list):
            # in this case, batch size = len(entities)
            assert isinstance(entities[0][0], int)
            entity_tensor = torch.tensor(entities, device=device).reshape(
                len(entities), -1
            )
        else:
            raise NotImplementedError(
                "higher order nested list is not supported"
            )
    elif isinstance(entities, torch.Tensor):
        assert entities.dim() == 2
        entity_tensor = entities.to(device)
    else:
        raise NotImplementedError("unsupported input entities type")
    return entity_tensor


def interleave_arange(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """
    Interleave the arange of two tensors
    Arguments:
        - start: [s1, ..., sn]
        - end: [e1, ..., en]
    Returns:
        - concatenation of
            [arange(s1, e1), arange(s2, e2), ..., arange(sn, en)]
    """
    assert start.size(0) == end.size(0)
    batch_size = start.size(0)
    diff = end - start
    max_len = diff.max().item()
    arange = torch.arange(max_len, device=start.device)
    brange = arange.unsqueeze(0).expand(batch_size, -1)
    mask = brange < diff.unsqueeze(1)

    offset_idx = start.unsqueeze(1) + arange
    return offset_idx[mask]


def refine_index(index_a: torch.Tensor, index_b: torch.Tensor) -> torch.Tensor:
    """
    index_a and index_b are two indices.
    Arguments:
        - index_a: index of each position
        - index_b: index of the second position
    return:
        - merged_index: satisfies that
            merged_index[i] = merged_index[j]
            if and only if
            index_a[i] = index_a[j] and index_b[i] = index_b[j]
    """

    assert index_a.size(0) == index_b.size(0)
    merged_index = torch.zeros_like(index_a)
    base = index_b.max().item() + 1
    combined = index_a * base + index_b
    _, merged_index = torch.unique(combined, return_inverse=True, sorted=True)
    return merged_index


class VariadicMatrix:
    """
    This class manages a list of sublists in 1d tensor
    data: the concated list of sublists.
    row_index: of the same shape of tensor.
        - row_index[i] is the index of sublists of data[i]
    global_index: the global index of the 1D data array and the row_index
    It can be used to select data.

    The operation functions, follows the practices of PyTorch.
    we use xxx_() to indicate the inplace operation.
    """

    data: torch.Tensor
    row_index: torch.Tensor

    cache = {}

    def __init__(
        self,
        data: torch.Tensor,
        row_index: torch.Tensor,
    ):
        assert data.size(0) == row_index.size(0)
        self.data = data
        self.row_index = row_index

    @property
    def offset(self):
        """
        The rowptr is used to index the data
        """
        offset = torch.tensor(
            [0] + torch.bincount(self.row_index).cumsum(0).tolist(),
            device=self.data.device,
        )
        assert offset[-1].item() == self.data.size(0)
        assert offset.size(0) == self.row_index.max().item() + 2
        return offset

    def to(self, device):
        self.data = self.data.to(device)
        self.row_index = self.row_index.to(device)
        return self

    def to_dense(self, padding_value: int = -1):
        """
        convert the variadic matrix into a dense matrix
        """
        sizes = self.offset[1:] - self.offset[:-1]
        flatten_sliced = torch.split(
            self.data, split_size_or_sections=sizes.tolist(), dim=0
        )
        dense_matrix = pad_sequence(
            flatten_sliced, batch_first=True, padding_value=padding_value
        )
        return dense_matrix

    @classmethod
    def from_dense(cls, matrix: torch.Tensor, padding_value: int = -1):
        """
        Instantiate a VariadicMatrix from a dense matrix
        """
        nrows, ncols = matrix.size()
        mask = matrix != padding_value
        data = matrix[mask]
        size_per_row = mask.sum(-1)
        row_index = torch.repeat_interleave(
            torch.arange(nrows), repeats=size_per_row
        )

        return cls(data=data, row_index=row_index)

    @classmethod
    def from_list(cls, nested_list):
        data = torch.tensor(list(chain(*nested_list)))
        row_index = torch.tensor(
            list(
                chain(
                    *[[i] * len(slist) for i, slist in enumerate(nested_list)]
                )
            )
        )
        return cls(data, row_index)

    def __getitem__(self, idx: int):
        return self.data[self.offset[idx] : self.offset[idx + 1]]

    def __len__(self):
        return self.row_index.max().item() + 1

    @property
    def size(self):
        assert self.row_index.size(0) == self.data.size(0)
        return self.row_index.size(0)

    def clone(self):
        return VariadicMatrix(self.data.clone(), self.row_index.clone())

    def validate(self):
        """
        Check the validity of the variadic matrix
        """
        assert self.row_index.size(0) == self.data.size(0)
        assert self.row_index.min().item() == 0
        unique_index = torch.unique(self.row_index)
        assert unique_index.size(0) == unique_index.max().item() + 1

    def sort_(self, descending: bool = False) -> None:
        """
        sort the data by index of two levels,
        the first level is the index of the sublist
        the second level is the value in each sublist
        """
        base = self.data.max().item() - self.data.min().item() + 1
        # the offset is used to make the global data unique
        offset = self.row_index * base
        if descending:
            offset = -offset
        sorted_index = torch.argsort(self.data + offset, descending=descending)
        # get the data by the indexing
        self.data = self.data[sorted_index]
        self.row_index = self.row_index[sorted_index]
        return sorted_index

    def unique_consecutive_(self, return_inverse: bool = False) -> None:
        """
        Make the sublist sorted and unique
        """
        base = self.data.max().item() - self.data.min().item() + 1
        # the offset is used to make the global data unique
        offset = self.row_index * base
        unique_data, inverse_index = torch.unique_consecutive(
            self.data + offset, return_inverse=True
        )
        unique_index = torch.scatter_reduce(
            input=torch.zeros_like(unique_data),
            dim=0,
            index=inverse_index,
            src=torch.arange(self.size, device=self.data.device),
            reduce="amin",
            include_self=False,
        )
        # get the data by indexing
        self.data = self.data[unique_index]
        self.row_index = self.row_index[unique_index]
        if return_inverse:
            return unique_index, inverse_index
        return unique_index

    def multiple_consecutive_(self) -> None:
        base = self.data.max().item() - self.data.min().item() + 1
        offset = self.row_index * base
        _, inverse_index = torch.unique_consecutive(
            self.data + offset, return_inverse=True
        )
        unique_index = torch.scatter_reduce(
            input=torch.zeros_like(self.row_index),
            dim=0,
            index=inverse_index,
            src=torch.arange(self.size, device=self.data.device),
            reduce="amin",
            include_self=False,
        )
        # populate the negative mask
        multiple_mask = torch.ones_like(self.data, dtype=torch.bool)
        multiple_mask[unique_index] = False
        self.data = self.data[multiple_mask]
        self.row_index = self.row_index[multiple_mask]
        return multiple_mask.nonzero().squeeze()

    def cartesian_product_with(
        self: "VariadicMatrix", other: "VariadicMatrix"
    ) -> Tuple["VariadicMatrix", "VariadicMatrix"]:
        """
        Calculate the cartesian product of two variadic matrices
        For sublists in the same position, calculate the cartesian product
        Then, merge the results
        """
        new_data = []
        new_index = []
        device = self.data.device
        for i in range(min(len(self), len(other))):
            row_a = self[i]
            row_b = other[i]
            if row_a.numel() == 0 or row_b.numel() == 0:
                continue
            cart = torch.cartesian_prod(row_a, row_b)
            new_data.append(cart)
            new_index.append(torch.full((cart.size(0),), i, device=device))

        if new_data:
            new_tensor = torch.cat(new_data, dim=0)
            new_index = torch.cat(new_index, dim=0)
            new_left = VariadicMatrix(new_tensor[:, 0], new_index)
            new_right = VariadicMatrix(new_tensor[:, 1], new_index)
            return new_left, new_right
        else:
            raise ValueError("No valid cartesian product found")

    def row_expand_as_(self, other: "VariadicMatrix") -> None:
        """
        This is like the above cartesian product, with the following assumption
        all sublists in self have only one element.
        Then we only need to repeat interleave the sublists and row indices.
        """

        batch_begin = other.offset[:-1]
        batch_end = other.offset[1:]
        batch_diff = batch_end - batch_begin
        self.data = torch.repeat_interleave(self.data, batch_diff)
        self.index = torch.repeat_interleave(batch_begin, batch_diff)

    def topk_index(self, k: int) -> Tuple["VariadicMatrix", torch.Tensor]:
        """
        Get the top k elements for each sublist
        This function will return new data, so we don't conduct inplace
        operation but make a clone first
        """
        # get the top k index
        row_start_gidx, row_end_gidx = self.offset[:-1], self.offset[1:]
        row_topk_end_gidx = torch.min(row_start_gidx + k, row_end_gidx)

        # compute top_k index in a smarter way
        topk_index = interleave_arange(row_start_gidx, row_topk_end_gidx)
        return topk_index

    def index_on_rows(
        self, index: torch.Tensor, reindex: bool = False
    ) -> "VariadicMatrix":
        """
        Index the variadic matrix by comparing elements in `index`
        against the `self.row_index`
        Arguments:
            index: the index of the rows
        Returns:
            the variadic matrix with the selected rows
        """

        # get the offset of the index
        offset = self.offset
        # get the offset of the index
        batch_start = offset[index]
        batch_end = offset[index + 1]

        # get the data
        data_select_index = interleave_arange(batch_start, batch_end)
        out_data = self.data[data_select_index]

        # new_row_index
        if reindex:
            new_row_idx = torch.arange(len(index), device=self.data.device)
        else:
            new_row_idx = index
        out_row_index = new_row_idx.repeat_interleave(batch_end - batch_start)

        return VariadicMatrix(out_data, out_row_index)

    def append(self, other: "VariadicMatrix") -> None:
        """
        Append the other variadic matrix to the current one
        """
        self.data = torch.cat([self.data, other.data], dim=0)
        self.row_index = torch.cat([self.row_index, other.row_index], dim=0)

    def index_global_(self, index: torch.Tensor) -> torch.Tensor:
        """
        Index the variadic matrix by the global index
        Arguments:
            index: the global index
        Returns:
            the selected data
        """
        self.data, self.row_index = self.data[index], self.row_index[index]

    def reindex_(self):
        """
        Reindex the variadic matrix
        specifically, compress the row_index
        """
        unique_index, self.row_index = torch.unique(
            self.row_index, return_inverse=True, sorted=True
        )

        return unique_index


def aggregation(
    score_tensor: torch.Tensor,
    assignment_vm: VariadicMatrix,
    reduce: str = "sum",
) -> Tuple[VariadicMatrix, VariadicMatrix]:
    """
    Aggregate the scores based on the assignment.

    The aggregation happens on the element

    Inputs:
        - score_tensor: the 1D tensor of scores
        - assignment_vm: the variadic matrix of assignments
            the internal 1D tensor is of the same shape as score_tensor
    Returns:
        - score_vm: the variadic matrix of scores
    """
    _, row_tail_index = assignment_vm.unique_consecutive_(return_inverse=True)
    reduced_score = scatter(
        src=score_tensor, index=row_tail_index, dim=0, reduce=reduce
    )

    score_vm = VariadicMatrix(
        data=reduced_score,
        row_index=assignment_vm.row_index,
    )

    return score_vm, assignment_vm
