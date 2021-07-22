import json
import torch
import torch.nn.functional as F
import numpy as np

SMALL_CONST = 1e-15
BIG_CONST = 1e10


def reverse_ids(input_ids, device):
    return torch.from_numpy(np.array(input_ids.tolist()[0][::-1])).unsqueeze(0).to(device)

def reverse_text(text):
    return ' '.join(reversed(text.split()))

def load_dataset(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def get_input_embeds(embedding, logits, o1_onehot=None, o2_onehot=None, device='cuda'):
    """
    embedding.shape = [50257, 1024]
    """
    probs = F.softmax(logits, dim=-1)
    if o1_onehot is not None:
        probs = torch.cat(
            (o1_onehot.type(torch.FloatTensor), probs.type(torch.FloatTensor)),
            dim=1)
    if o2_onehot is not None:
        probs = torch.cat(
            (probs.type(torch.FloatTensor), o2_onehot.type(torch.FloatTensor)),
            dim=1)
    probs = probs.to(device)
    return torch.matmul(probs, embedding.weight)


def top_k_filter(logits, k, probs=False, device='cuda'):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.

    Args:
        probs (bool): Whether `logits` is indeed probabilities
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


def get_token_from_logits(logits, temperature=1.0, top_k=1):
    """
    logits.shape = [batch_size]
    """
    # normalize
    logits = top_k_filter(logits, k=top_k)
    probs = F.softmax(logits, dim=-1)

    # greedy
    _, last = torch.topk(probs, k=1, dim=-1)

    return last


def get_text_from_logits(logits, tokenizer, temperature=1.0, top_k=1):
    output_so_far = None
    for i in range(logits.shape[1]):
        last = get_token_from_logits(logits[:, i, :], temperature, top_k)

        # update context/output_so_far appending the new token
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)

    text = tokenizer.decode(output_so_far.tolist()[0])
    text = text.replace('\n', ' ')

    return text
