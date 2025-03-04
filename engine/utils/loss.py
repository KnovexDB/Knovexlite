import torch

import torch.nn.functional as F

def loss_softmax(logits, answers):
    """
    Compute the training loss.
    """
    boolean_target = torch.zeros_like(logits)
    for i, t in enumerate(answers):
        boolean_target[i, t] = 1
    pred_pos = logits * boolean_target
    max_n = torch.max(pred_pos, dim=-1)[0].unsqueeze(-1)
    loss = -F.log_softmax(logits - max_n, dim=-1)[boolean_target.bool()]
    loss = loss.mean()
    return loss