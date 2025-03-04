import logging

import torch
from torch_geometric.utils import index_to_mask

from engine.utils.data import VariadicMatrix

logger = logging.getLogger(__name__)


def first_unique_idx(tensor1d: torch.Tensor):
    """
    Given one 1D tensor, find the first index of each unique value
    """
    uval, inv_idx = torch.unique(tensor1d, sorted=True, return_inverse=True)
    first_idx = torch.scatter_reduce(
        input=uval.new_zeros(len(uval)),
        dim=0,
        index=inv_idx,
        src=torch.arange(len(inv_idx), device=tensor1d.device),
        reduce="amin",
        include_self=False,
    )
    return first_idx


def mask_propagate(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
    direction: str = "source_to_target",
):
    """
    This function propagates the mask from the source nodes to the target nodes

    Input:
        - edge_index: torch.Tensor, shape [2, E]
        - mask: torch.Tensor, shape [N]
    Output:
        - propagated_mask: torch.Tensor, shape [N]
    """
    if direction == "source_to_target":
        from_edge_index = edge_index[0]
        to_edge_index = edge_index[1]
    elif direction == "target_to_source":
        from_edge_index = edge_index[1]
        to_edge_index = edge_index[0]
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # edge_mask[i] = 1 if the source node of edge i is masked by mask
    edge_mask = node_mask[from_edge_index].bool()
    # node_id_dup[i] = the target node of edge i if the source node is masked
    node_id_dup = to_edge_index[edge_mask]
    output_mask = torch.zeros_like(node_mask)
    output_mask.index_fill_(0, node_id_dup, 1)
    return output_mask


def bfs(edge_index: torch.Tensor, source_mask: torch.Tensor):
    """
    This function finds the BFS traversal order of the graph

    Input:
        - edge_index: torch.Tensor, shape [2, E]
        - source_mask: torch.Tensor, shape [N]
    Output:
        - bfs_mask_list: List[torch.Tensor], each tensor is of shape [N]
            bfs_mask_list[0] = source_mask
            bfs_mask_list[i] = mask of the i-th layer of BFS traversal
    """
    visited = source_mask.bool().clone()
    frontier = visited.clone()
    bfs_mask_list = [visited.float()]

    while frontier.any():
        neighbor_mask = mask_propagate(edge_index, frontier)
        frontier = neighbor_mask & ~visited

        visited = visited | frontier
        bfs_mask_list.append(frontier)

    return bfs_mask_list


def topological_order(
    edge_index: torch.Tensor, source_mask: torch.Tensor
) -> VariadicMatrix:
    """
    This function finds the topological order of the graph

    Input:
        - edge_index: torch.Tensor, shape [2, E]
        - source_mask: torch.Tensor, shape [N], it is the constant mask.
    Output:
        - topological_order: torch.Tensor, shape [N]
    """
    _visited = source_mask.bool().clone()
    _edge_index = edge_index.clone()

    topological_order = [source_mask.nonzero().squeeze().tolist()]

    while _visited.logical_not().any():
        _edge_mask_from_visited = _visited[_edge_index[0]]
        _edge_index_from_visited = _edge_index[:, _edge_mask_from_visited]
        # find the first edge from each source node
        idx_to_expand = first_unique_idx(_edge_index_from_visited[0])
        targets = _edge_index_from_visited[1, idx_to_expand].unique(sorted=True)
        topological_order.append(targets.tolist())
        # then, expand the targets to the visited nodes
        _visited[targets] = True
        # remove the used edges from _edge_index
        to_mask = index_to_mask(targets, size=source_mask.size(0))
        used_edge_mask = _edge_mask_from_visited & to_mask[_edge_index[1]]
        _edge_index = _edge_index[:, used_edge_mask.logical_not()]

    topological_order_vm = VariadicMatrix.from_list(topological_order)
    return topological_order_vm.to(source_mask.device)
