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
from general_utils import flatten_json, normalize_text, build_embedding, load_glove_vocab, pre_proc, get_context_span, feature_gen, token2id

parser = argparse.ArgumentParser(
    description='Preprocessing test files.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--no_answer', action='store_true',
                    help='do not have ground truth answer.')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')

args = parser.parse_args()
test_file = 'SQuAD/dev-v1.1.json'
wv_file = args.wv_file
wv_dim = args.wv_dim
nlp = spacy.load('en', parser=False)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))

glove_vocab = load_glove_vocab(wv_file, wv_dim) # return a "set" of vocabulary
log.info('glove loaded.')

#===========================================================
#=================== Work on test data =====================
#===========================================================

def proc_test(article):
    rows = []
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            id_, question = qa['id'], qa['question']
            if args.no_answer:
                answers = ['', '', '']
            else:
                answers = qa['answers']
                answers = [a['text'] for a in answers]
            rows.append((id_, context, question, answers))
    return rows

test = flatten_json(test_file, proc_test)
test = pd.DataFrame(test, columns=['id', 'context', 'question', 'answers'])
log.info('test json data flattened.')

testC_iter = (pre_proc(c) for c in test.context)
testQ_iter = (pre_proc(q) for q in test.question)
testC_docs = [doc for doc in nlp.pipe(
    testC_iter, batch_size=64, n_threads=args.threads)]
testQ_docs = [doc for doc in nlp.pipe(
    testQ_iter, batch_size=64, n_threads=args.threads)]
testC_unnorm_tokens = [[w.text for w in doc] for doc in testC_docs]
log.info('unnormalized tokens for test is obtained.')

test['context_span'] = [get_context_span(a, b) for a, b in zip(test.context, testC_unnorm_tokens)]
log.info('context span for test is generated.')

testC_tags, testC_ents, testC_features = feature_gen(testC_docs, testQ_docs, args.no_match)
log.info('features for test is generated.')

with open('SQuAD/train_meta.msgpack', 'rb') as f:
    meta = msgpack.load(f, encoding='utf8')
tr_vocab = meta['vocab']
def build_test_vocab(questions, contexts): # most vocabulary comes from tr_vocab
    existing_vocab = set(tr_vocab)
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab and w in glove_vocab]))
    vocab = tr_vocab + new_vocab
    log.info('train vocab {0}, total vocab {1}'.format(len(tr_vocab), len(vocab)))
    return vocab

# tokens
testC_tokens = [[normalize_text(w.text) for w in doc] for doc in testC_docs]
testQ_tokens = [[normalize_text(w.text) for w in doc] for doc in testQ_docs]
test_vocab = build_test_vocab(testQ_tokens, testC_tokens) # tr_vocab is a subset of test_vocab
testC_ids = token2id(testC_tokens, test_vocab, unk_id=1)
testQ_ids = token2id(testQ_tokens, test_vocab, unk_id=1)
# tags
vocab_tag = list(nlp.tagger.tag_names)
testC_tag_ids = token2id(testC_tags, vocab_tag) # vocab_tag same as training
# entities
vocab_ent = [''] + nlp.entity.cfg[u'actions']['1']
testC_ent_ids = token2id(testC_ents, vocab_ent) # vocab_ent same as training
log.info('vocabulary for test is built.')

test_embedding = build_embedding(wv_file, test_vocab, wv_dim)
# tr_embedding is a submatrix of test_embedding
log.info('got embedding matrix for test.')

# don't store row name in csv
test.to_csv('SQuAD/test.csv', index=False, encoding='utf8')

meta = {
    'vocab': test_vocab,
    'embedding': test_embedding.tolist()
}
with open('SQuAD/test_meta.msgpack', 'wb') as f:
    msgpack.dump(meta, f)

result = {
    'question_ids': testQ_ids,
    'context_ids': testC_ids,
    'context_features': testC_features, # exact match, tf
    'context_tags': testC_tag_ids, # POS tagging
    'context_ents': testC_ent_ids, # Entity recognition
}
with open('SQuAD/test_data.msgpack', 'wb') as f:
    msgpack.dump(result, f)

log.info('saved test to disk.')
