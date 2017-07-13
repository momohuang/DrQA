import re
import os
import sys
import random
import string
import logging
import argparse
from os.path import basename
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import pandas as pd
from drqa.model import LEGOReaderModel
from general_utils import score, BatchGen, _exact_match

parser = argparse.ArgumentParser(
    description='Predict using a Lego Reader model.'
)
parser.add_argument('-o', '--output', default='',
                    help='path for the output of model prediction.')
parser.add_argument('-m', '--model', default='',
                    help='testing model pathname, e.g. "models/checkpoint_epoch_11.pt"')
parser.add_argument('--test_meta', default='SQuAD/test_meta.msgpack',
                    help='path to preprocessed testing meta file.')
parser.add_argument('--test_data', default='SQuAD/test_data.msgpack',
                    help='path to preprocessed testing data file.')
parser.add_argument('-bs', '--batch_size', default=32)
parser.add_argument('--show', type=int, default=10)
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')

args = parser.parse_args()
if args.model == '':
    print("model file is not provided")
    sys.exit(-1)
if args.model[-3:] != '.pt':
    print("does not recognize the model file")
    sys.exit(-1)
args.output = 'pred-' + basename(args.model)[:-3] + '.txt'

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)

def main():
    log.info('[program starts.]')
    checkpoint = torch.load(args.model)
    opt = checkpoint['config']
    opt['cuda'] = args.cuda
    opt['seed'] = args.seed
    opt['do_inter_att'] = True
    state_dict = checkpoint['state_dict']
    log.info('[model loaded.]')

    test, test_embedding, questions, test_answer = load_test_data(opt)
    model = LEGOReaderModel(opt, state_dict = state_dict)
    log.info('[Data loaded.]')

    model.setup_eval_embed(test_embedding)

    if args.cuda:
        model.cuda()

    batches = BatchGen(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    predictions = []
    for batch in batches:
        predictions.extend(model.predict(batch))
    em, f1 = score(predictions, test_answer)
    log.warning("Test EM: {} F1: {}".format(em, f1))

    wrong_pred = []
    with open(args.output, 'w', encoding='utf8') as f:
        for i, pred in enumerate(predictions):
            f.write(pred+'\n')
            if _exact_match(pred, test_answer[i]) == False:
                wrong_pred.append((i, pred))
    error_samples = random.sample(wrong_pred, args.show)
    for (i, pred) in error_samples:
        print('Context: ', test[i][-2])
        print('Question: ', questions[i], '\n')
        print('Answers: ', test_answer[i])
        print('Predictions: ', pred)
        print('\n==================\n')

def load_test_data(opt):
    with open(args.test_meta, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    assert opt['embedding_dim'] == embedding.size(1)

    with open(args.test_data, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    data_orig = pd.read_csv('SQuAD/test.csv')

    assert opt['num_features'] == len(data['context_features'][0][0])
    span = data_orig['context_span'].tolist()
    test = list(zip(
        data['context_ids'],
        data['context_features'],
        data['context_tags'],
        data['context_ents'],
        data['question_ids'],
        data_orig['context'].tolist(),
        [eval(x) for x in span],
    ))

    assert len(test) == len(data_orig['answers'].tolist())
    test_answer = data_orig['answers'].tolist()
    test_answer = [eval(ans) for ans in test_answer] # ans is str, eval(ans) is list
    return test, embedding, data_orig['question'].tolist(), test_answer # test_answer may be a dummy variable

if __name__ == '__main__':
    main()
