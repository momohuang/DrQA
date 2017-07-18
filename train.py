import re
import os
import sys
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import pandas as pd
import numpy as np
from drqa.model import LEGOReaderModel
from general_utils import score, BatchGen

parser = argparse.ArgumentParser(
    description='Train a Lego Reader model.'
)
# system
parser.add_argument('--name', default='', help='additional name of the current run')
parser.add_argument('--log_file', default='output.log',
                    help='path for log file.')
parser.add_argument('--log_per_updates', type=int, default=20,
                    help='log model loss per x updates (mini-batches).')
parser.add_argument('--train_meta', default='SQuAD/train_meta.msgpack',
                    help='path to preprocessed training meta file.')
parser.add_argument('--train_data', default='SQuAD/train_data.msgpack',
                    help='path to preprocessed training data file.')
parser.add_argument('--dev_meta', default='SQuAD/dev_meta.msgpack',
                    help='path to preprocessed validation meta file.')
parser.add_argument('--dev_data', default='SQuAD/dev_data.msgpack',
                    help='path to preprocessed validation data file.')
parser.add_argument('--model_dir', default='models',
                    help='path to store saved models.')
parser.add_argument('--save_last_only', action='store_true',
                    help='only save the final models.')
parser.add_argument('--eval_per_epoch', type=int, default=1,
                    help='perform evaluation per x epoches.')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
# training
parser.add_argument('-e', '--epoches', type=int, default=20)
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-rs', '--resume', default='',
                    help='previous model pathname. '
                         'e.g. "models/checkpoint_epoch_11.pt"')
parser.add_argument('-ro', '--resume_options', action='store_true',
                    help='use previous model options, ignore the cli and defaults.')
parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                    help='reduce initial (resumed) learning rate by this factor.')
parser.add_argument('-op', '--optimizer', default='adamax',
                    help='supported optimizer: adamax, sgd')
parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                    help='only applied to SGD.')
parser.add_argument('-mm', '--momentum', type=float, default=0,
                    help='only applied to SGD.')
parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                    help='finetune top-x embeddings.')
parser.add_argument('--fix_embeddings', action='store_true',
                    help='if true, `tune_partial` will be ignored.')
parser.add_argument('--rnn_padding', action='store_true',
                    help='perform rnn padding (much slower but more accurate).')
# model
parser.add_argument('--question_merge', default='self_attn')
parser.add_argument('--doc_layers', type=int, default=3)
parser.add_argument('--question_layers', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--pos', type=bool, default=True)
parser.add_argument('--pos_size', type=int, default=56,
                    help='how many kinds of POS tags.')
parser.add_argument('--pos_dim', type=int, default=12,
                    help='the embedding dimension for POS tags.')
parser.add_argument('--ner', type=bool, default=True)
parser.add_argument('--ner_size', type=int, default=19,
                    help='how many kinds of named entity tags.')
parser.add_argument('--ner_dim', type=int, default=8,
                    help='the embedding dimension for named entity tags.')
parser.add_argument('--no_wvec_align', dest='wvec_align', action='store_false')

parser.add_argument('--gated_input', dest='gated_input', action='store_true')
parser.add_argument('--gated_int_att_input', action='store_true')

parser.add_argument('--do_C2Q', action='store_true')
parser.add_argument('--do_coattention', action='store_true')

parser.add_argument('--inter_att_type', default='relu_FC')
parser.add_argument('--inter_att_concat', default='concat')

parser.add_argument('--do_multi_att', action='store_true')
parser.add_argument('--multi_att_do_relu', action='store_true')
parser.add_argument('--multi_att_key', type=int, default=128)
parser.add_argument('--multi_att_val', type=int, default=128)
parser.add_argument('--multi_att_h', type=int, default=6)
parser.add_argument('--multi_att_dropout', type=float, default=0)

parser.add_argument('--concat_rnn_layers', type=bool, default=True)
parser.add_argument('--dropout_emb', type=float, default=0.3)
parser.add_argument('--dropout_rnn', type=float, default=0.3)
parser.add_argument('--dropout_rnn_output', type=bool, default=True)
parser.add_argument('--max_len', type=int, default=15)
parser.add_argument('--rnn_type', default='lstm',
                    help='supported types: rnn, gru, lstm')

args = parser.parse_args()

if args.name != '':
    args.model_dir = args.model_dir + '_' + args.name
    args.log_file = os.path.dirname(args.log_file) + 'output_' + args.name + '.log'

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

def main():
    log.info('[program starts.]')
    opt = vars(args) # changing opt will change args
    train, train_embedding, opt = load_train_data(opt)
    dev, dev_embedding, dev_answer = load_dev_data(opt)
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(args.resume)
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = LEGOReaderModel(opt, train_embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        for i in range(checkpoint['epoch']):
            random.shuffle(list(range(len(train))))  # synchronize random seed
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
    else:
        model = LEGOReaderModel(opt, train_embedding)
        epoch_0 = 1

    model.setup_eval_embed(dev_embedding)

    if args.cuda:
        model.cuda()

    if args.resume:
        batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        predictions = []
        for batch in batches:
            predictions.extend(model.predict(batch))
        em, f1 = score(predictions, dev_answer)
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        best_val_score = f1
    else:
        best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + args.epoches):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        # eval
        if epoch % args.eval_per_epoch == 0:
            batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
            predictions = []
            for batch in batches:
                predictions.extend(model.predict(batch))
            em, f1 = score(predictions, dev_answer)
            log.warning("Epoch {} - dev EM: {} F1: {}".format(epoch, em, f1))
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epoches - 1:
            model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch)
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    os.path.join(model_dir, model_file),
                    os.path.join(model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    log.info('[learning rate reduced by {}]'.format(lr_decay))
    return optimizer


def load_train_data(opt):
    with open(args.train_meta, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)

    with open(args.train_data, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    data_orig = pd.read_csv('SQuAD/train.csv')

    opt['num_features'] = len(data['context_features'][0][0])
    span = data_orig['context_span'].tolist()
    train = list(zip( # list() due to lazy evaluation of zip
        data['context_ids'],
        data['context_features'],
        data['context_tags'],
        data['context_ents'],
        data['question_ids'],
        data_orig['answer_start_token'].tolist(),
        data_orig['answer_end_token'].tolist(),
        data_orig['context'].tolist(),
        [eval(x) for x in span]
    ))
    return train, embedding, opt

def load_dev_data(opt): # can be extended to true test set
    with open(args.dev_meta, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    assert opt['embedding_dim'] == embedding.size(1)

    with open(args.dev_data, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    data_orig = pd.read_csv('SQuAD/dev.csv')

    assert opt['num_features'] == len(data['context_features'][0][0])
    span = data_orig['context_span'].tolist()
    dev = list(zip(
        data['context_ids'],
        data['context_features'],
        data['context_tags'],
        data['context_ents'],
        data['question_ids'],
        data_orig['context'].tolist(),
        [eval(x) for x in span]
    ))

    assert len(dev) == len(data_orig['answers'].tolist())
    dev_answer = data_orig['answers'].tolist()
    dev_answer = [eval(ans) for ans in dev_answer] # ans is str, eval(ans) is list
    return dev, embedding, dev_answer

if __name__ == '__main__':
    main()
