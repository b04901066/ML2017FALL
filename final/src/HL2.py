import os, re, sys, csv, json, numpy, pandas, difflib
from collections import OrderedDict
from random import *

window_size = 30

# readin
# testing_label.json  list.len=1450  
test_label_f = open( sys.argv[1], 'r')
test_label   = json.load(test_label_f)
test_label_f.close()

with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow([ 'id', 'answer'])
    for paragra in range( len(test_label['data']) ):
        for conte in range( len(test_label['data'][paragra]['paragraphs']) ):
            context = test_label['data'][paragra]['paragraphs'][conte]['context'].replace('\n', 'N')

            for qas in range( len(test_label['data'][paragra]['paragraphs'][conte]['qas']) ):
                answer_score = []
                question = test_label['data'][paragra]['paragraphs'][conte]['qas'][qas]['question'].replace('\n', 'N')
                
                for ans in range( len(context)-window_size ):
                    '''
                    for qq in range(len(qas_jieba)):
                        for words in range(len(answer_list_jieba[ans])):
                            if qas_jieba[qq] == answer_list_jieba[ans][words]:
                                answer_score[ans] += 1
                    '''
                    answer_score.append(difflib.SequenceMatcher(None,
                                                                context[ans : ans + window_size],
                                                                question ).ratio())

                start_point = numpy.argmax( numpy.asarray(answer_score, dtype=float) )
                output = ''
                for out in range(start_point, start_point + window_size):
                    if question.find( context[out] ) == (-1):
                        output += str(out)
                        output += ' '
                output = output[:-1]
                ID = test_label['data'][paragra]['paragraphs'][conte]['qas'][qas]['id']
                spamwriter.writerow([ ID, output])
