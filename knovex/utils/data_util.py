from itertools import chain
from typing import Union, List

import torch
from torch.nn.utils.rnn import pad_sequence

name2lstr = {
    "1p": "r1(s1,f)",
    "2p": "r1(s1,e1)&r2(e1,f)",  # 2p
    "3p": "r1(s1,e1)&r2(e1,e2)&r3(e2,f)",  # 3p
    "2i": "r1(s1,f)&r2(s2,f)",  # 2i
    "3i": "r1(s1,f)&r2(s2,f)&r3(s3,f)",  # 3i
    "ip": "r1(s1,e1)&r2(s2,e1)&r3(e1,f)",  # ip
    "pi": "r1(s1,e1)&r2(e1,f)&r3(s2,f)",  # pi
    "2in": "r1(s1,f)&!r2(s2,f)",  # 2in
    "3in": "r1(s1,f)&r2(s2,f)&!r3(s3,f)",  # 3in
    "inp": "r1(s1,e1)&!r2(s2,e1)&r3(e1,f)",  # inp
    "pin": "r1(s1,e1)&r2(e1,f)&!r3(s2,f)",  # pin
    "pni": "r1(s1,e1)&!r2(e1,f)&r3(s2,f)",  # pni
    "2u": "r1(s1,f)|r2(s2,f)",  # 2u
    "up": "(r1(s1,e1)|r2(s2,e1))&r3(e1,f)",  # up
    "2u-dm": "!(!r1(s1,f)&!r2(s2,f))",  # 2u-dm
    "up-dm": "!(!r1(s1,e1)|r2(s2,e1))&r3(e1,f)",  # up-dm
}


newlstr2name = {  # new naming convention: m for multi edge, a for anchor node, c for circle
    '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))': 'pni',
    '((r1(s1,e1))&(r2(e1,f)))&(r3(e1,f))': '2m',
    '((r1(s1,e1))&(r2(e1,f)))&(!(r3(e1,f)))': '2nm',
    '(((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f)))&(r4(e1,e2))': '3mp',
    '(((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f)))&(r4(e2,f))': '3pm',
    '(((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f)))&(r4(e1,f))': 'im',
    # '(((r1(s1,e1))&(r2(e1,f)))&(r3(e1,f)))&(r4(s2,f))': 'mi',
    '(r1(s1,f))&(r2(e1,f))': '2il',
    # '(r1(e1,f))&(!(r2(s1,f)))': '2ln',
    '((r1(s1,f))&(r2(s2,f)))&(r3(e1,f))': '3il',
    '((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2))': '3c',
    '(((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2)))&(r6(e1,f))': '3cm',
    # '(((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2)))&(r6(e1,e2))': '3mc',
    '(((((r1(s1,e1))&(r2(e1,e3)))&(r3(s2,e2)))&(r4(e2,e3)))&(r5(e1,e2)))&(r6(e3,f))': '3pcp'
}

index2EFOX_minimal = {
    0: '((r1(s1,f1))&(r2(s2,f2)))&(r3(f1,f2))',
    1: '((r1(s1,e1))&(r2(e1,f1)))&(r3(e1,f2))'
}

index2newlstr = {
    0: '((r1(s1,e1))&(r2(e1,f)))&(r3(e1,f))',
    1: '((r1(s1,e1))&(r2(e1,f)))&(!(r3(e1,f)))',
    2: '(((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f)))&(r4(e1,e2))',
    3: '(((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f)))&(r4(e2,f))',
    4: '(((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f)))&(r4(e1,f))',
    5: '(r1(s1,f))&(r2(e1,f))',
    6: '(r1(e1,f))&(!(r2(s1,f)))',
    7: '((r1(s1,f))&(r2(s2,f)))&(r3(e1,f))',
    8: '((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2))',
    9: '(((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2)))&(r6(e1,f))',
    10: '(((((r1(s1,e1))&(r2(e1,e3)))&(r3(s2,e2)))&(r4(e2,e3)))&(r5(e1,e2)))&(r6(e3,f))'
}

def _iter_triple_from_tsv(triple_file, to_int, check_size):
    with open(triple_file, 'rt') as f:
        for line in f.readlines():
            triple = line.strip().split()
            if check_size:
                assert len(triple) == check_size
            if to_int:
                triple = [int(t) for t in triple]
            yield triple


def iter_triple_from_tsv(triple_files, to_int: bool=True, check_size: int=3):
    if isinstance(triple_files, list):
        return chain(*[iter_triple_from_tsv(tfile) for tfile in triple_files])
    elif isinstance(triple_files, str):
        return _iter_triple_from_tsv(triple_files, to_int, check_size)
    else:
        raise NotImplementedError("invalid input of triple files")


def tensorize_batch_entities(
        entities: Union[List[int], List[List[int]], torch.Tensor],
        device) -> torch.Tensor:
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
            entity_tensor = torch.tensor(
                entities, device=device).reshape(-1, 1)
        elif isinstance(entities[0], list):
            # in this case, batch size = len(entities)
            assert isinstance(entities[0][0], int)
            entity_tensor = torch.tensor(
                entities, device=device).reshape(len(entities), -1)
        else:
            raise NotImplementedError(
                "higher order nested list is not supported")
    elif isinstance(entities, torch.Tensor):
        assert entities.dim() == 2
        entity_tensor = entities.to(device)
    else:
        raise NotImplementedError("unsupported input entities type")
    return entity_tensor


class RaggedBatch:
    def __init__(self, flatten, sizes):
        self.flatten = flatten
        self.sizes = sizes

    def run_ops_on_flatten(self, opfunc):
        return RaggedBatch(
            flatten=opfunc(self.flatten),
            sizes=self.sizes)

    def to_dense_matrix(self, padding_value):
        # split the first axis of the flattened Tensor by sizes
        flatten_sliced = torch.split(
            self.flatten, split_size_or_sections=self.sizes, dim=0)
        dense_matrix = pad_sequence(
            flatten_sliced, batch_first=True, padding_value=padding_value)
        # if the self.flattened is of shape [L, *]
        # then dense_matrix is of shape [batch_size, max_of_self.sizes, *]
        return dense_matrix