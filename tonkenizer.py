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

    prev_output_tokens.masked_fill_(prev_output_tokens == -100, torch.tensor(pad_token_id))

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
            self.id2token = self.special_tokens[:]

            for w in sorted(self.word2fre.keys(), key=lambda x: self.word2fre[w][::-1]):
                f = self.word2fre[w]

                if f >= self.min_freq:
                    self.id2token.append(f)

            self.token2id = {v: i for i, v in enumerate(self.id2token)}
            self.pad_id = self.token2id['<pad>']
            self.eos_id = self.token2id['</s>']
            self.unk_id = self.token2id['<unk>']
            self.sos = self.token2id['<s>']
            self.ignore = self.pad_id
        elif self.level == "sentencepiece":
            self.tokenizer = MBartTokenizer(**tokenizer_cfg)

            self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
            self.ignore_id = self.pad_id
            with open(tokenizer_cfg['pruneids_file'], 'r') as f:
                self.pruneids = pickle.load(f)
                for t in ["<pad>", "<s>", "</s>", "<unk>"]:
                    _id = self.tokenizer.convert_tokens_to_ids(t)
                    assert self.pruneids[_id] == _id, '{}->{}'.format(_id, self.pruneids[_id])
                self.pruneids_reverse = {v:k for k, v in self.pruneids.items()}
                self.lang_id = self.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]
                self.sos_id = self.lang_id
                self.eos_id = self.pruneids[self.tokenizer.convert_tokens_to_ids('</s>')]
        else:
            raise ValueError

    def generate_decoder_labels(self, input_ids):
        decoder_labels = torch.where(
            input_ids == self.lang_index,  # already be mapped into pruned_vocab
            torch.ones_like(input_ids) * self.ignore_index, input_ids)
        return decoder_labels

    def generate_decoder_inputs(self, input_ids):
        decoder_inputs = shift_tokens_right(input_ids,
                                            pad_token_id=self.pad_id,
                                            ignore_index=self.pad_id)
        return decoder_inputs

    def ids_to_prune(self, input_ids):
        output_prunes = []

        for seq in input_ids:
            prune_seq = [self.pruneids[_id if _id in self.pruneids
                            else self.tokenizer.convert_tokens_to_ids("<unk>")]
                         for _id in seq]
            output_prunes.append(prune_seq)
        output_prunes = torch.tensor(output_prunes, dtype=torch.long)
        return output_prunes

    def prune_to_ids(self, input_prune):
        batch_size, max_len = input_prune.shape
        output_ids = input_prune.clone()

        for b in range(batch_size):
            for i in range(max_len):
                _id = output_ids[b, i].item()
                if _id not in self.pruneids_reverse:
                    new_id = self.tokenizer.convert_tokens_to_ids("<unk>")
                else:
                    new_id = self.pruneids_reverse[_id]
                output_ids[b, i] = new_id

        return output_ids

    def batch_decode(self, sequences):
        sequences = sequences[:, 1:]
        if self.level == "sentencepiece":
            sequences = self.prune_to_ids(sequences)
            decode_seqs = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            if "de" in self.tokenizer.tgt_lang:
                for di, d in enumerate(decode_seqs):
                    d = d[:-1] + " ."
                    decode_seqs[di] = d
        elif self.level == "word":
            decode_seqs = [" ".join([s for s in seq]) for seq in sequences]

        else:
            raise ValueError

        return decode_seqs

    def __call__(self, input_str):
        if self.level == "sentencepiece":
            with self.tokenizer.as_target_tokenizer():
                raw_outputs = self.tokenizer(input_str, return_attention_mask=True,
                                             return_length=True, padding='longest')
            outputs = {}
            prune_inputs = self.ids_to_prune(raw_outputs['input_ids'])
            outputs['label'] = self.generate_decoder_labels(prune_inputs)
            outputs['decode_input_ids'] = self.generate_decoder_inputs(prune_inputs)

        elif self.level == "word":
            batch_labels, batch_decoder_input_ids, batch_lenghts = [[]]*3
            for _str in input_str:
                label =[]
                input_ids =[self.sos_id]
                for i, s in _str.split():
                    _id = self.token2id[s]
                    label.append(_id)
                    input_ids.append(_id)
                label.append(self.eos_id)

                batch_labels.append(label)
                batch_decoder_input_ids.append(input_ids)
                batch_lenghts.append(len(label))

            max_len = max(batch_lenghts)
            padded_batch_labels, padded_batch_decoder_input_ids = [], []

            for label, decoder_input_ids in zip(batch_labels, batch_decoder_input_ids):
                pad_label = label + [self.pad_id]*(max_len - len(label))
                pad_decoder_ids = decoder_input_ids + [self.ignore_id]*(max_len - len(decoder_input_ids))

                assert  len(pad_label) == len(pad_decoder_ids), f"{len(pad_label)} is not equal to {len(pad_decoder_ids)}"
                padded_batch_labels.append(pad_label)
                padded_batch_decoder_input_ids.append(pad_decoder_ids)
            outputs = {
                "labels" : torch.tensor(padded_batch_labels),
                "decoder_input_ids" : torch.tensor(padded_batch_decoder_input_ids)
            }

        else:
            raise ValueError

        return outputs


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
            seq += (maxlen - len_seq) * [pad_id]
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
            gls_ids = self.tokens_to_ids(seq.split())
            batch_gls_lengths.append(len(gls_ids))

            gls_ids = self.pad_seq(seq=gls_ids, maxlen=max_len, pad_id=self.pad_id)
            batch_gls_ids.append(gls_ids)

        batch_gls_ids = torch.tensor(batch_gls_ids)
        batch_gls_lengths = torch.tensor(batch_gls_lengths)

        return {"gls_len": batch_gls_lengths, "gls_ids": batch_gls_ids}


class TokenizerGlossToText(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)

        self.src_lang = tokenizer_cfg['src_lang']

    def __call__(self, batch_seq):
        max_len = max([len(gls.split()) for gls in batch_seq]) + 2

        batch_gls_ids = []

        attentions_mask = torch.zeros(len(batch_seq), max_len, dtype=torch.long)

        for i, seq in enumerate(batch_seq):
            gls_ids = self.tokens_to_ids(seq.split())
            gls_ids = gls_ids + [self.gloss2id['</s>'], self.gloss2id[self.src_lang]]

            attentions_mask[i, :len(gls_ids)] = 1

            gls_ids = self.pad_seq(seq=gls_ids, maxlen=max_len, pad_id=self.gloss2id['<pad>'])
            batch_gls_ids.append(gls_ids)

        batch_gls_ids = torch.tensor(batch_gls_ids)

        return {"input_ids": batch_gls_ids, "attention_mask": attentions_mask}