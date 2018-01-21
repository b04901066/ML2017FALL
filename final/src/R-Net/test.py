import datetime
import json
import os
import shutil
import sys
import time

import torch
from torch import optim
from torch.autograd import Variable

import r_net as RNet

from tqdm import tqdm

from HL2 import rule_base

class Tester(object):
    def __init__(self, args, dataloader_test, char_embedding_config, word_embedding_config,
                 sentence_encoding_config, pair_encoding_config, self_matching_config, pointer_config):

        self.dataloader_test = dataloader_test

        self.model = RNet.Model(args, char_embedding_config, word_embedding_config, sentence_encoding_config,
                                pair_encoding_config, self_matching_config, pointer_config)
        self.start_time = datetime.datetime.now().strftime('%b-%d_%H-%M')

        print('loading model {}'.format(args.model))
        checkpoint = torch.load(args.model)
        self.model.load_state_dict(checkpoint['state_dict'])

        if torch.cuda.is_available():
            self.model = self.model.cuda(args.device_id)
        else:
            self.model = self.model.cpu()

    def test(self):
        self.model.eval()
        pred_result_text = {}
        pred_result = {}
        for _, batch in tqdm(enumerate(self.dataloader_test)):
            question_ids, questions, passages, passage_tokenized, raw_passage, raw_question = batch
            questions.variable(volatile=True)
            passages.variable(volatile=True)
            begin_, end_ = self.model(questions, passages)  # batch x seq

            _, pred_begin = torch.max(begin_, 1)
            _, pred_end = torch.max(end_, 1)

            pred = torch.stack([pred_begin, pred_end], dim=1)

            for i, (begin, end) in enumerate(pred.cpu().data.numpy()):
                ans = passage_tokenized[i][begin:end+1]
                qid = question_ids[i]
                if len(ans) == 0:
                    #print('evaluating {} by HL!'.format(qid))
                    output, output_text = rule_base(raw_passage[i], raw_question[i])
                    pred_result[qid] = output[:]
                    pred_result_text[qid] = output_text[:]
                else:
                    real_begin = len(''.join(passage_tokenized[i][0:begin]))
                    real_end = real_begin + len(''.join(ans))
                    pred_result[qid] = ' '.join(str(x) for x in list(range(real_begin, real_end)))
                    pred_result_text[qid] = ''.join(ans)
        self.model.train()

        save_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        fout = open('output-{}.csv'.format(save_time), 'w')
        fout.write('id,answer\n')
        for qid, ans in tqdm(pred_result.items()):
            fout.write('{},{}\n'.format(qid, ans))
        fout.close()

        fout = open('text-{}.csv'.format(save_time), 'w')
        fout.write('id,answer\n')
        for qid, ans in tqdm(pred_result_text.items()):
            fout.write('{},{}\n'.format(qid, ans))
        fout.close()

        return
