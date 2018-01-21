import array
import json
import os
import random
from collections import Counter

import torch

import jieba
jieba.dt.cache_file = 'jieba.cache.final'

def jieba_word_tokenize(sent):
    l = jieba.lcut(sent, cut_all=False)
    return l

def load_word_vectors(root, wv_type, dim):
    '''

    From https://github.com/pytorch/text/

    BSD 3-Clause License

    Copyright (c) James Bradbury and Soumith Chintala 2016,
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    '''
    '''Load word vectors from a path, trying .pt, .txt, and .zip extensions.'''
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    #fname = os.path.join(root, wv_type + '.' + dim)
    fname = os.path.join(root, wv_type)
    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        return torch.load(fname_pt)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        print('load', fname_txt)
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, 'r') as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors %s from %s' % (wv_type, root))

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        print('Loading word vectors from {}'.format(fname_txt))
        for line in trange(len(cm)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret

class RawExample(object):
    pass

def read_test_json(path):
    with open(path) as fin:
        data = json.load(fin)
    examples = []

    for topic in data['data']:
        title = topic['title']
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']

            for qa in qas:
                question = qa['question']
                question_id = qa['id']

                e = RawExample()
                e.title = title
                e.passage = context
                e.question = question
                e.question_id = question_id
                examples.append(e)

    return examples

def truncate_word_counter(word_counter, max_symbols):
    words = [(freq, word) for word, freq in word_counter.items()]
    words.sort()
    return {word: freq for freq, word in words[:max_symbols]}

def read_embedding(root, word_type, dim):
    wv_dict, wv_vectors, wv_size = load_word_vectors(root, word_type, dim)
    return wv_dict, wv_vectors, wv_size

def get_rnn(rnn_type):
    rnn_type = rnn_type.lower()
    if rnn_type == 'gru':
        network = torch.nn.GRU
    elif rnn_type == 'lstm':
        network = torch.nn.LSTM
    else:
        raise ValueError('Invalid RNN type %s' % rnn_type)
    return network

def sort_idx(seq):
    '''
    :param seq: variable
    :return:
    '''
    return sorted(range(seq.size(0)), key=lambda x: seq[x])
