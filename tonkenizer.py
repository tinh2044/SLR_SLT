import pickle
import torch
import json
from collections import defaultdict
from transformers import MBartTokenizer


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, ignore_index: int=-100) -> torch.Tensor:
    """
    This function shifts tokens to the right in a tensor, handling padding and ignoring specific tokens.
    
    :param input_ids: `input_ids` is a PyTorch tensor containing token IDs
    :type input_ids: torch.Tensor
    :param pad_token_id: The `pad_token_id` parameter is the token ID used to represent padding tokens
    in the input tensor. It is used to identify tokens that are added to the input sequence to make all
    sequences in a batch of the same length during processing
    :type pad_token_id: int
    :param ignore_index: The `ignore_index` parameter is used to specify a token index that should be
    ignored during token shifting operations. In the provided function `shift_tokens_right`, this
    parameter is set to a default value of `-100` if not explicitly provided by the caller. The function
    uses this `ignore_index` value
    :type ignore_index: int
    :return: The function `shift_tokens_right` returns a torch.Tensor with the input tokens shifted one
    position to the right.
    """

    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id is has to be define"

    prev_output_tokens.masked_fill_(prev_output_tokens==ignore_index, torch.tensor(pad_token_id))

    idx_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)

    for i, ip in enumerate(idx_eos.squeeze(1)):
        input_ids[i, ip:] = ignore_index

    decoder_start_tokens = prev_output_tokens.gather(1, idx_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, -1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens
    return prev_output_tokens

class BaseTokenizer(object):
    def __init__(self, tokenizer_cfg):
        self.tokenizer_cfg = tokenizer_cfg
    
    def __call__():
        pass