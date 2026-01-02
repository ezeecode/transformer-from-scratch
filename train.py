import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilingualDataset, causal_mask
from model import build_transformer


def get_all_sentences(dataset, lang):
        for item in dataset:
            yield item['translation'][lang]


# Get or build tokenizer
def get_or_build_tokenizer(config, dataset, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang=lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    dataset_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    # split dataset into training and validation
    train_dataset_size = int(0.9 * len(dataset_raw))
    val_dataset_size = len(dataset_raw) - train_dataset_size

    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_size, val_dataset_size])

    train_dataset = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, 
                                     config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    validation_dataset = BilingualDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, 
                                         config['lang_src'], config['lang_tgt'], config['seq_len'])
    

    # check max sequence lengths in raw dataset
    max_len_src = max([len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids) for item in dataset_raw])
    max_len_tgt = max([len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids) for item in dataset_raw])

    print(f"Max source sequence length: {max_len_src}")
    print(f"Max target sequence length: {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt




def get_model(config, vocab_src_len, vocab_tgt_len):
     
     model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
     return model

