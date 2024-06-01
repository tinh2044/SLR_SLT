import pickle
import torch
import json
from collections import defaultdict
from transformers import MBartTokenizer


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, ignore_index: int = -100) -> torch.Tensor:
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

    prev_output_tokens.masked_fill_(prev_output_tokens == ignore_index, torch.tensor(pad_token_id))

    idx_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)

    for i, ip in enumerate(idx_eos.squeeze(1)):
        input_ids[i, ip:] = ignore_index

    decoder_start_tokens = prev_output_tokens.gather(1, idx_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, -1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens
    return prev_output_tokens


class TextTokenizer:
    def __init__(self, tokenizer_cfg: dict):
        super(TextTokenizer, self).__init__(tokenizer_cfg)

        self.level = tokenizer_cfg.get("level", "sentencepiece")

        if self.level == "level":
            self.min_freq = tokenizer_cfg.get("min_freq", 0)
            with open(tokenizer_cfg['tokenizer_file'], 'r') as f:
                tokenizer_info = json.load(f)

            self.word2fre = tokenizer_info['word2fre']
            self.special_tokens = tokenizer_info['special_tokens']

            for w in sorted(self.word2fre.keys(), key=lambda x: self.word2fre[w][::-1]):
                pass


class BaseTokenizer:

    def __init__(self, tokenizer_cfg: dict):
        # super().__init__(tokenizer_cfg)
        self.tokenizer_cfg = tokenizer_cfg
        with open(tokenizer_cfg['gloss_file'], 'r') as f:
            self.gloss2id = pickle.load(f)

        self.gloss2id = defaultdict(lambda: self.gloss2id['<unk>'], self.gloss2id)
        self.id2gloss = {_id: _gloss for _id, _gloss in self.gloss2id}

        self.is_lowercase = tokenizer_cfg.get("is_lowercase", True)

    def tokens_to_ids(self, tokens):
        if type(tokens) is list:
            return [self.gloss2id[x.lower() if self.is_lowercase else x] for x in tokens]
        else:
            return self.gloss2id[tokens]

    def ids_to_tokens(self, ids):
        if type(ids) is list:
            return [self.id2gloss[x] for x in ids]
        else:
            return self.id2gloss[ids]

    def pad_seq(self, seq, maxlen, pad_id):
        len_seq = len(seq)
        if len_seq < maxlen:
            seq += (maxlen - len_seq)*[pad_id]
        return seq
    def __len__(self):
        return len(self.gloss2id)


class TokenizerSignToGloss(BaseTokenizer):
    def __int__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        if "<s>" in self.gloss2id:
            self.start_token = "<s>"
            self.start_id = self.tokens_to_ids("<s>")
        elif "<si>" in self.gloss2id:
            self.start_token = "<si>"
            self.start_id = self.tokens_to_ids("<si>")
        else:
            raise ValueError

        assert self.start_id == 0, self.start_id

        self.pad_token = "<pad>"
        self.pad_id = self.tokens_to_ids("<pad>")

    def __call__(self, batch_seq):
        # get max length of a batch sequence
        max_len = max([len(seq.split()) for seq in batch_seq])

        batch_gls_lengths, batch_gls_ids = [], []

        for i, seq in enumerate(batch_seq):

            # convert a sentence to list of its id
            gls_ids = self.tokens_to_ids([x for x in seq.split()])
            batch_gls_lengths.append(len(gls_ids))

            gls_ids = self.pad_seq(seq=gls_ids, maxlen=max_len, pad_id=self.pad_id)
            batch_gls_ids.append(gls_ids)

        batch_gls_ids = torch.tensor(batch_gls_ids)
        batch_gls_lengths = torch.tensor(batch_gls_lengths)

        return {"gls_len": batch_gls_lengths, "gls_ids": batch_gls_ids}





