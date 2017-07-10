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

parser = argparse.ArgumentParser(
    description='Preprocessing data files, about 10 minutes to run.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words. '
                         'Otherwise consider question words first.')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')

args = parser.parse_args()
trn_file = 'SQuAD/train-v1.1.json'
dev_file = 'SQuAD/dev-v1.1.json'
wv_file = args.wv_file
wv_dim = args.wv_dim
nlp = spacy.load('en', parser=False)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))

#===============================================================
#=================== Work on training data =====================
#===============================================================

def flatten_json(file, proc_func):
    with open(file, encoding="utf8") as f:
        data = json.load(f)['data']
    rows = []
    for i in range(len(data)):
        rows.append(proc_func(data[i]))
    rows = sum(rows, [])
    return rows

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

def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '

def pre_proc(text):
    # make hyphens, spaces clean
    text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

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
            log.info("Something wrong with get_span()")
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

def feature_gen(C_docs, Q_docs):
    C_tags = [[w.tag_ for w in doc] for doc in C_docs]
    C_ents = [[w.ent_type_ for w in doc] for doc in C_docs]
    C_features = []
    for question, context in zip(Q_docs, C_docs):
        counter_ = collections.Counter(w.text.lower() for w in context)
        total = sum(counter_.values())
        term_freq = [counter_[w.text.lower()] / total for w in context]
        if args.no_match:
            C_features.append(list(zip(term_freq)))
        else:
            question_word = {w.text for w in question}
            question_lower = {w.text.lower() for w in question}
            question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
            match_origin = [w.text in question_word for w in context]
            match_lower = [w.text.lower() in question_lower for w in context]
            match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context]
            C_features.append(list(zip(term_freq, match_origin, match_lower, match_lemma)))
    return C_tags, C_ents, C_features

trC_tags, trC_ents, trC_features = feature_gen(trC_docs, trQ_docs)
log.info('features for training is generated.')

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def build_train_vocab(questions, contexts): # vocabulary will also be sorted accordingly
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter], key=counter.get, reverse=True)
    else:
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys()],
                        key=counter.get, reverse=True)
    total = sum(counter.values())
    log.info('vocab#: {0}, token#: {1}'.format(len(vocab), total))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab

def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids

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

def build_train_embedding(embed_file, targ_vocab, dim_vec):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, dim_vec))
    emb[0] = 0 # <PAD> should be all 0 (using broadcast)

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

tr_embedding = build_train_embedding(wv_file, tr_vocab, wv_dim)
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

def get_dev_span(context, context_token):
    p_str = 0
    p_token = 0
    t_span = []
    while p_str < len(context):
        if re.match('\s', context[p_str]):
            p_str += 1
            continue

        token = context_token[p_token]
        token_len = len(token)
        if context[p_str:p_str + token_len] != token:
            log.info("Something wrong with get_span()")
            return []
        t_span.append((p_str, p_str + token_len))

        p_str += token_len
        p_token += 1
    return t_span

dev['context_span'] = [get_dev_span(a, b) for a, b in zip(dev.context, devC_unnorm_tokens)]
initial_len = len(dev)
log.info('context span for dev is generated.')

devC_tags, devC_ents, devC_features = feature_gen(devC_docs, devQ_docs)
log.info('features for dev is generated.')

def build_dev_vocab(questions, contexts): # most vocabulary comes from tr_vocab
    existing_vocab = set(tr_vocab)
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab]))
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

def build_dev_embedding(embed_file, targ_vocab, dim_vec):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, dim_vec))
    emb[0] = 0 # <PAD> should be all 0 (using broadcast)

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

dev_embedding = build_dev_embedding(wv_file, dev_vocab, wv_dim)
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
