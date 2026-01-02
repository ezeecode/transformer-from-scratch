from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: Any) -> Any:
        src_tgt_pair = self.dataset[index]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        encoder_input_tokens = self.tokenizer_src.encode(src_text).ids
        decoder_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        encoder_num_padding_tokens = self.seq_len - len(encoder_input_tokens) - 2 # +2 for SOS and EOS
        decoder_num_padding_tokens = self.seq_len - len(decoder_input_tokens) - 1 # we add SOS only to decoder input, EOS is added to target 

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError(f"Sequence length {self.seq_len} is too small for the given sentence pair: {src_text} ||| {tgt_text}")
        
        # add SOS and EOS tokens and padding to source text
        encoder_input = torch.cat([
            self.sos_token,
            torch.Tensor(encoder_input_tokens, dytype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(encoder_num_padding_tokens)
        ])

        # add SOS token and padding to decoder input
        decoder_input = torch.cat([
            self.sos_token,
            torch.Tensor(decoder_input_tokens, dtype=torch.int64),
            self.pad_token.repeat(decoder_num_padding_tokens)
        ])

        # create target tokens by adding EOS token and padding (what we expect the decoder to output)
        target = torch.cat([
            torch.Tensor(decoder_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(decoder_num_padding_tokens)
        ])

        # sanity checks for sequence lengths
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert target.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,                                                     # {seq_len}
            "decoder_input": decoder_input,                                                     # {seq_len}
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # {1, 1, seq_len} -- 1 for non-pad tokens, 0 for pad tokens
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            # {1, 1, seq_len, seq_len} -- 1 for non-pad tokens in allowed positions, 0 otherwise
            "target": target,                                                                    # {seq_len}
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    

def causal_mask(size: int) -> torch.Tensor:
    """ Creates a causal mask for the decoder self-attention mechanism.
        The mask prevents the attention mechanism from attending to future tokens.
        The mask is a lower triangular matrix of shape (1, size, size) with 1s in the allowed positions and 0s elsewhere.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) # upper triangular matrix with 1s above the diagonal -- (1, size, size)
    return mask == 0  # convert to boolean mask





    



