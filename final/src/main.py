import torch
import os
import argparse

from utils import read_embedding
from dataset import DeltaDataset
from test import Tester

def read_vocab(vocab_config):
    wv_dict, wv_vectors, wv_size = read_embedding(vocab_config['embedding_root'],
                                                  vocab_config['embedding_type'],
                                                  vocab_config['embedding_dim'])

    embed_size = wv_vectors.size(1)

    itos = vocab_config['specials'][:]
    stoi = {}

    itos.extend(list(w for w, i in sorted(wv_dict.items(), key=lambda x: x[1])))

    for idx, word in enumerate(itos):
        stoi[word] = idx

    vectors = torch.zeros([len(itos), embed_size])

    for word, idx in stoi.items():
        if word not in wv_dict or word in vocab_config['specials']:
            continue
        vectors[idx, :wv_size].copy_(wv_vectors[wv_dict[word]])
    return itos, stoi, vectors


def main():
    parser = argparse.ArgumentParser(description='PyTorch R-net')

    parser.add_argument('model', type=str, help='model file name')
    parser.add_argument('test_json', type=str, help='path to test-v1.1.json')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--device_id', type=int, default=None, help='CUDA device id')

    args = parser.parse_args()

    word_vocab_config = {
        '<UNK>': 0,
        '<PAD>': 1,
        '<start>': 2,
        '<end>': 3,
        'insert_start': '<SOS>',
        'insert_end': '<EOS>',
        'tokenization': 'jieba',
        'specials': ['<UNK>', '<PAD>', '<SOS>', '<EOS>'],
        'embedding_root': os.path.join('embedding', 'word'),
        'embedding_type': 'word_vectors',
        'embedding_dim': 50
    }
    print('Reading Vocab')
    char_vocab_config = word_vocab_config.copy()
    char_vocab_config['embedding_root'] = os.path.join('embedding', 'char')
    char_vocab_config['embedding_type'] = 'char_vectors'

    itos, stoi, wv_vec = read_vocab(word_vocab_config)
    itoc, ctoi, cv_vec = read_vocab(char_vocab_config)

    char_embedding_config = {'embedding_weights': cv_vec,
                             'padding_idx': word_vocab_config['<UNK>'],
                             'update': True,
                             'bidirectional': True,
                             'cell_type': 'gru', 'output_dim': 300}

    word_embedding_config = {'embedding_weights': wv_vec,
                             'padding_idx': word_vocab_config['<UNK>'],
                             'update': False}

    sentence_encoding_config = {'hidden_size': 55,
                                'num_layers': 3,
                                'bidirectional': True,
                                'dropout': 0.2}

    pair_encoding_config = {'hidden_size': 55,
                            'num_layers': 3,
                            'bidirectional': True,
                            'dropout': 0.2,
                            'gated': True, 'mode': 'GRU',
                            'rnn_cell': torch.nn.GRUCell,
                            'attn_size': 55,
                            'residual': False}

    self_matching_config = {'hidden_size': 55,
                            'num_layers': 3,
                            'bidirectional': True,
                            'dropout': 0.2,
                            'gated': True, 'mode': 'GRU',
                            'rnn_cell': torch.nn.GRUCell,
                            'attn_size': 55,
                            'residual': False}

    pointer_config = {'hidden_size': 55,
                      'num_layers': 3,
                      'dropout': 0.2,
                      'residual': False,
                      'rnn_cell': torch.nn.GRUCell}

    test_json = args.test_json

    test = read_dataset(test_json, itos, stoi, itoc, ctoi)
    test_dataloader = test.get_dataloader(args.batch_size)

    tester = Tester(args, test_dataloader,
                      char_embedding_config, word_embedding_config,
                      sentence_encoding_config, pair_encoding_config,
                      self_matching_config, pointer_config)
    tester.test()


def read_dataset(json_file, itos, stoi, itoc, ctoi):
    print('Reading dataset')
    dataset = DeltaDataset(json_file, itos, stoi, itoc, ctoi)
    return dataset

if __name__ == '__main__':
    main()
