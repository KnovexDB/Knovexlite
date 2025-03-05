import logging
from collections import defaultdict
import torch

from src.language.efo_lang import parse_lstr_to_lformula

beta_lstr_list = [
    "r1(s1,f)",  # 1p
    "r1(s1,e1)&r2(e1,f)",  # 2p
    "r1(s1,e1)&r2(e1,e2)&r3(e2,f)",  # 3p
    "r1(s1,f)&r2(s2,f)",  # 2i
    "r1(s1,f)&r2(s2,f)&r3(s3,f)",  # 3i
    "r1(s1,e1)&r2(s2,e1)&r3(e1,f)",  ## ip
    "r1(s1,e1)&r2(e1,f)&r3(s2,f)",  # pi
    "r1(s1,f)&!r2(s2,f)",  # 2in
    "r1(s1,f)&r2(s2,f)&!r3(s3,f)",  # 3in
    "r1(s1,e1)&!r2(s2,e1)&r3(e1,f)",  # inp
    "r1(s1,e1)&r2(e1,f)&!r3(s2,f)",  # pin
    "r1(s1,e1)&!r2(e1,f)&r3(s2,f)",  # pni
    "r1(s1,f)|r2(s2,f)",  # 2u
    "(r1(s1,e1)|r2(s2,e1))&r3(e1,f)",  # up
    "(r1(s1,e1)&r3(e1,f))|(r2(s2,e1)&r3(e1,f))",  # up-dnf
]

beta_names = [
    "1p",
    "2p",
    "3p",
    "2i",
    "3i",
    "ip",
    "pi",
    "2in",
    "3in",
    "inp",
    "pin",
    "pni",
    "2u",
    "up",
    "up-dnf",
]


lstr2name_betae = {}
for s, n in zip(beta_lstr_list, beta_names):
    lstr2name_betae[parse_lstr_to_lformula(s).lstr()] = n

# new naming convention: m for multi edge, a for anchor node, c for circle
lstr2name_efo1 = {
    "((r1(s1,e1))&(r2(e1,f)))&(r3(e1,f))": "2m",
    "((r1(s1,e1))&(r2(e1,f)))&(!(r3(e1,f)))": "2nm",
    "(((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f)))&(r4(e1,e2))": "3mp",
    "(((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f)))&(r4(e2,f))": "3pm",
    "(((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f)))&(r4(e1,f))": "im",
    # '(((r1(s1,e1))&(r2(e1,f)))&(r3(e1,f)))&(r4(s2,f))': 'mi',
    "(r1(s1,f))&(r2(e1,f))": "2il",
    # '(r1(e1,f))&(!(r2(s1,f)))': '2ln',
    "((r1(s1,f))&(r2(s2,f)))&(r3(e1,f))": "3il",
    "((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2))": "3c",
    "(((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2)))&(r6(e1,f))": "3cm",
    # '(((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2)))&(r6(e1,e2))': '3mc',
    "(((((r1(s1,e1))&(r2(e1,e3)))&(r3(s2,e2)))&(r4(e2,e3)))&(r5(e1,e2)))&(r6(e3,f))": "3pcp",
    "((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))": "pni",
}

lstr2name = {**lstr2name_betae, **lstr2name_efo1}
name2lstr = {v: k for k, v in lstr2name.items()}


def evaluate_metric(score, hard_target, easy_target, lstr_list):
    """
    args:
        score: [B, n_k]
        easy_target: [B, n_k]
        hard_target: [B, n_k]
    """
    score = score.detach()
    ranked_entity_ids = score.argsort(dim=-1, descending=True)
    entity_rankings = ranked_entity_ids.argsort(dim=-1, descending=False)

    batch_metric = defaultdict(lambda: defaultdict(list))

    for i in range(entity_rankings.shape[0]):
        ranking = entity_rankings[i]

        if easy_target is None or len(easy_target[i]) == 0:
            # if there is no easy answer, skip
            easy_answer_rank = None
        else:
            easy_answer_rank = ranking[easy_target[i]].view(-1, 1)

        if len(hard_target[i]) == 0:
            # if there is no hard answer, skip
            raise ValueError("There is no hard answer!")
        else:
            hard_answer_rank = ranking[hard_target[i]].view(-1, 1)

        # remove better easy answers from hard answers' rankings
        if easy_answer_rank is not None:
            hard_answer_rank -= torch.sum(
                hard_answer_rank.view(-1, 1) > easy_answer_rank.view(1, -1),
                dim=-1,
                keepdim=True,
            )
        # remove better hard answers from hard answers' ranking
        hard_answer_rank -= torch.sum(
            hard_answer_rank.view(-1, 1) > hard_answer_rank.view(1, -1),
            dim=-1,
            keepdim=True,
        )

        key = lstr2name[lstr_list[i]]

        mrr = (1 / (1 + hard_answer_rank)).float().mean().item()
        hit1 = (hard_answer_rank < 1).float().mean().item()
        hit3 = (hard_answer_rank < 3).float().mean().item()
        hit10 = (hard_answer_rank < 10).float().mean().item()

        batch_metric[key]["MRR"].append(mrr)
        batch_metric[key]["Hit1"].append(hit1)
        batch_metric[key]["Hit3"].append(hit3)
        batch_metric[key]["Hit10"].append(hit10)

    return batch_metric
