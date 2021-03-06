import re
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import pandas as pd
import argparse
import collections
import multiprocessing
import logging
import random
from general_utils import flatten_json, normalize_text, build_embedding, load_glove_vocab, pre_proc, get_context_span, feature_gen, token2id

parser = argparse.ArgumentParser(
    description='Preprocessing train + dev files, about 10 minutes to run.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words.'
                         'Otherwise consider question words first.')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, embedding init, etc.')


args = parser.parse_args()
trn_file = 'SQuAD/train-v1.1.json'
dev_file = 'SQuAD/dev-v1.1.json'
wv_file = args.wv_file
wv_dim = args.wv_dim
nlp = spacy.load('en', parser=False)

random.seed(args.seed)
np.random.seed(args.seed)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))

glove_vocab = load_glove_vocab(wv_file, wv_dim) # return a "set" of vocabulary
log.info('glove loaded.')

#===============================================================
#=================== Work on training data =====================
#===============================================================

def proc_train(article):
    rows = []
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            assert len(answers) == 1
            answer = answers[0]['text']
            answer_start = answers[0]['answer_start']
            answer_end = answer_start + len(answer)
            rows.append((id_, context, question, answer, answer_start, answer_end))
    return rows

train = flatten_json(trn_file, proc_train)
train = pd.DataFrame(train,
                     columns=['id', 'context', 'question', 'answer',
                              'answer_start', 'answer_end'])
log.info('train json data flattened.')

trC_iter = (pre_proc(c) for c in train.context)
trQ_iter = (pre_proc(q) for q in train.question)
trC_docs = [doc for doc in nlp.pipe(
    trC_iter, batch_size=64, n_threads=args.threads)]
trQ_docs = [doc for doc in nlp.pipe(
    trQ_iter, batch_size=64, n_threads=args.threads)]
trC_unnorm_tokens = [[w.text for w in doc] for doc in trC_docs]
log.info('unnormalized tokens for training is obtained.')

def get_train_span(context, answer, context_token, answer_start, answer_end):
    p_str = 0
    p_token = 0
    t_start, t_end, t_span = -1, -1, []
    while p_str < len(context):
        if re.match('\s', context[p_str]):
            p_str += 1
            continue

        token = context_token[p_token]
        token_len = len(token)
        if context[p_str:p_str + token_len] != token:
            log.info("Something wrong with get_train_span()")
            return (None, None, [])
        t_span.append((p_str, p_str + token_len))

        if (p_str <= answer_start and answer_start < p_str + token_len):
            t_start = p_token
        if (p_str < answer_end and answer_end <= p_str + token_len):
            t_end = p_token

        p_str += token_len
        p_token += 1
    #print(context, answer, context[max(0, answer_start - 3):min(len(context), answer_end + 3)])
    if t_start == -1 or t_end == -1:
        return (None, None, [])
    else:
        return (t_start, t_end, t_span)

train['answer_start_token'], train['answer_end_token'], train['context_span'] = \
    zip(*[get_train_span(a, b, c, d, e) for a, b, c, d, e in
          zip(train.context, train.answer, trC_unnorm_tokens,
              train.answer_start, train.answer_end)])
initial_len = len(train)
train.dropna(inplace=True) # modify self DataFrame
log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(train), initial_len))
log.info('answer span for training is generated.')

trC_tags, trC_ents, trC_features = feature_gen(trC_docs, trQ_docs, args.no_match)
log.info('features for training is generated.')

def build_train_vocab(questions, contexts): # vocabulary will also be sorted accordingly
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    else:
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in glove_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in glove_vocab],
                        key=counter.get, reverse=True)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab {1}/{0} OOV {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab

# tokens
trC_tokens = [[normalize_text(w.text) for w in doc] for doc in trC_docs]
trQ_tokens = [[normalize_text(w.text) for w in doc] for doc in trQ_docs]
tr_vocab = build_train_vocab(trQ_tokens, trC_tokens)
trC_ids = token2id(trC_tokens, tr_vocab, unk_id=1)
trQ_ids = token2id(trQ_tokens, tr_vocab, unk_id=1)
# tags
vocab_tag = list(nlp.tagger.tag_names)
trC_tag_ids = token2id(trC_tags, vocab_tag)
# entities
vocab_ent = [''] + nlp.entity.cfg[u'actions']['1']
trC_ent_ids = token2id(trC_ents, vocab_ent)
log.info('Found {} POS tags.'.format(len(vocab_tag)))
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
log.info('vocabulary for training is built.')

tr_embedding = build_embedding(wv_file, tr_vocab, wv_dim)
log.info('got embedding matrix for training.')

# don't store row name in csv
train.to_csv('SQuAD/train.csv', index=False, encoding='utf8')

meta = {
    'vocab': tr_vocab,
    'embedding': tr_embedding.tolist()
}
with open('SQuAD/train_meta.msgpack', 'wb') as f:
    msgpack.dump(meta, f)

result = {
    'question_ids': trQ_ids,
    'context_ids': trC_ids,
    'context_features': trC_features, # exact match, tf
    'context_tags': trC_tag_ids, # POS tagging
    'context_ents': trC_ent_ids, # Entity recognition
}
with open('SQuAD/train_data.msgpack', 'wb') as f:
    msgpack.dump(result, f)

log.info('saved training to disk.')

#==========================================================
#=================== Work on dev data =====================
#==========================================================

def proc_dev(article):
    rows = []
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            answers = [a['text'] for a in answers]
            rows.append((id_, context, question, answers))
    return rows

dev = flatten_json(dev_file, proc_dev)
dev = pd.DataFrame(dev, columns=['id', 'context', 'question', 'answers'])
log.info('dev json data flattened.')

devC_iter = (pre_proc(c) for c in dev.context)
devQ_iter = (pre_proc(q) for q in dev.question)
devC_docs = [doc for doc in nlp.pipe(
    devC_iter, batch_size=64, n_threads=args.threads)]
devQ_docs = [doc for doc in nlp.pipe(
    devQ_iter, batch_size=64, n_threads=args.threads)]
devC_unnorm_tokens = [[w.text for w in doc] for doc in devC_docs]
log.info('unnormalized tokens for dev is obtained.')

dev['context_span'] = [get_context_span(a, b) for a, b in zip(dev.context, devC_unnorm_tokens)]
log.info('context span for dev is generated.')

devC_tags, devC_ents, devC_features = feature_gen(devC_docs, devQ_docs, args.no_match)
log.info('features for dev is generated.')

def build_dev_vocab(questions, contexts): # most vocabulary comes from tr_vocab
    existing_vocab = set(tr_vocab)
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab and w in glove_vocab]))
    vocab = tr_vocab + new_vocab
    log.info('train vocab {0}, total vocab {1}'.format(len(tr_vocab), len(vocab)))
    return vocab

# tokens
devC_tokens = [[normalize_text(w.text) for w in doc] for doc in devC_docs]
devQ_tokens = [[normalize_text(w.text) for w in doc] for doc in devQ_docs]
dev_vocab = build_dev_vocab(devQ_tokens, devC_tokens) # tr_vocab is a subset of dev_vocab
devC_ids = token2id(devC_tokens, dev_vocab, unk_id=1)
devQ_ids = token2id(devQ_tokens, dev_vocab, unk_id=1)
# tags
devC_tag_ids = token2id(devC_tags, vocab_tag) # vocab_tag same as training
# entities
devC_ent_ids = token2id(devC_ents, vocab_ent) # vocab_ent same as training
log.info('vocabulary for dev is built.')

dev_embedding = build_embedding(wv_file, dev_vocab, wv_dim)
# tr_embedding is a submatrix of dev_embedding
log.info('got embedding matrix for dev.')

# don't store row name in csv
dev.to_csv('SQuAD/dev.csv', index=False, encoding='utf8')

meta = {
    'vocab': dev_vocab,
    'embedding': dev_embedding.tolist()
}
with open('SQuAD/dev_meta.msgpack', 'wb') as f:
    msgpack.dump(meta, f)

result = {
    'question_ids': devQ_ids,
    'context_ids': devC_ids,
    'context_features': devC_features, # exact match, tf
    'context_tags': devC_tag_ids, # POS tagging
    'context_ents': devC_ent_ids, # Entity recognition
}
with open('SQuAD/dev_data.msgpack', 'wb') as f:
    msgpack.dump(result, f)

log.info('saved dev to disk.')
