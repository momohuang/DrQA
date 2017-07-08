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
parser.add_argument('--wv_cased', type=bool, default=True,
                    help='treat the words as cased or not.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words. '
                         'Otherwise consider question words first.')
parser.add_argument('--sample_size', type=int, default=0,
                    help='size of sample data (for debugging).')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')

args = parser.parse_args()
trn_file = 'SQuAD/train-v1.1.json'
dev_file = 'SQuAD/dev-v1.1.json'
wv_file = args.wv_file
wv_dim = args.wv_dim

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing...')


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def load_glove_vocab(file):
    vocab = set()
    with open(file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            vocab.add(token)
    return vocab

glove_vocab = load_glove_vocab(wv_file) # return a "set" of vocabulary
log.info('glove loaded.')

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

def proc_dev(article):
    rows = []
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            answers = [a['text'] for a in answers]
            rows.append((id_, context, question, answers))
    return rows

train = flatten_json(trn_file, proc_train)
train = pd.DataFrame(train,
                     columns=['id', 'context', 'question', 'answer',
                              'answer_start', 'answer_end'])
dev = flatten_json(dev_file, proc_dev)
dev = pd.DataFrame(dev,
                   columns=['id', 'context', 'question', 'answers'])
log.info('json data flattened.')

nlp = spacy.load('en', parser=False, tagger=False, entity=False)


def pre_proc(text):
    # make spaces clean
    text = re.sub('\s+', ' ', text)
    return text

context_iter = (pre_proc(c) for c in train.context)
context_tokens = [[w.text for w in doc] for doc in nlp.pipe(
    context_iter, batch_size=64, n_threads=multiprocessing.cpu_count())]
log.info('got intial tokens.')


def get_answer_index(context, answer, context_token, answer_start, answer_end):
    p_str = 0
    p_token = 0
    while p_str < len(context):
        if re.match('\s', context[p_str]):
            p_str += 1
            continue
        token = context_token[p_token]
        token_len = len(token)
        if context[p_str:p_str + token_len] != token:
            return (None, None)
        if p_str == answer_start:
            t_start = p_token
        p_str += token_len
        if p_str == answer_end:
            try:
                return (t_start, p_token)
            except UnboundLocalError as e:
                return (None, None)
        p_token += 1
    return (None, None)

train['answer_start_token'], train['answer_end_token'] = \
    zip(*[get_answer_index(a, b, c, d, e) for a, b, c, d, e in
          zip(train.context, train.answer, context_tokens,
              train.answer_start, train.answer_end)])
initial_len = len(train)
train.dropna(inplace=True) # modify self DataFrame
log.info('drop {} inconsistent samples.'.format(initial_len - len(train)))
log.info('answer pointer generated.')

questions = list(train.question) + list(dev.question)
contexts = list(train.context) + list(dev.context)

nlp = spacy.load('en')
context_text = [pre_proc(c) for c in contexts]
question_text = [pre_proc(q) for q in questions]
question_docs = [doc for doc in nlp.pipe(
    iter(question_text), batch_size=64, n_threads=multiprocessing.cpu_count())]
context_docs = [doc for doc in nlp.pipe(
    iter(context_text), batch_size=64, n_threads=multiprocessing.cpu_count())]
if args.wv_cased:
    question_tokens = [[normalize_text(w.text) for w in doc] for doc in question_docs]
    context_tokens = [[normalize_text(w.text) for w in doc] for doc in context_docs]
else:
    question_tokens = [[normalize_text(w.text).lower() for w in doc] for doc in question_docs]
    context_tokens = [[normalize_text(w.text).lower() for w in doc] for doc in context_docs]
context_token_span = [[(w.idx, w.idx + len(w.text)) for w in doc] for doc in context_docs]
context_tags = [[w.tag_ for w in doc] for doc in context_docs]
context_ents = [[w.ent_type_ for w in doc] for doc in context_docs]
context_features = []
if not args.no_match:
    for question, context in zip(question_docs, context_docs):
        question_word = {w.text for w in question}
        question_lower = {w.text.lower() for w in question}
        question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
        match_origin = [w.text in question_word for w in context]
        match_lower = [w.text.lower() in question_lower for w in context]
        match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context]
        context_features.append(list(zip(match_origin, match_lower, match_lemma))) # zip returns generator, so we need to evaluate it using list()
log.info('tokens generated')


def build_vocab(questions, contexts): # vocabulary will also be sorted accordingly
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
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
    return vocab, counter


def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids

vocab, counter = build_vocab(question_tokens, context_tokens)
# tokens
question_ids = token2id(question_tokens, vocab, unk_id=1)
context_ids = token2id(context_tokens, vocab, unk_id=1)

# term frequency in document
context_tf = []
for doc in context_tokens:
    counter_ = collections.Counter(w.lower() for w in doc)
    total = sum(counter_.values())
    context_tf.append([counter_[w.lower()] / total for w in doc])
if context_features != []:
    context_features = [[list(w) + [tf] for w, tf in zip(doc, tfs)] for doc, tfs in
                        zip(context_features, context_tf)]
else:
    context_features = [[[tf] for tf in tfs] for tfs in context_tf]

# tags
vocab_tag = list(nlp.tagger.tag_names)
context_tag_ids = token2id(context_tags, vocab_tag)
# entities, build dict on the fly
counter_ent = collections.Counter(w for doc in context_ents for w in doc)
vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
log.info('Found {} POS tags.'.format(len(vocab_tag)))
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
context_ent_ids = token2id(context_ents, vocab_ent)
log.info('vocab built.')


def build_embedding(embed_file, targ_vocab, dim_vec):
    vocab_size = len(targ_vocab)
    emb = np.zeros((vocab_size, dim_vec))
    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

embedding = build_embedding(wv_file, vocab, wv_dim)
log.info('got embedding matrix.')

# don't store row name in csv
train.to_csv('SQuAD/train.csv', index=False, encoding='utf8')
dev.to_csv('SQuAD/dev.csv', index=False, encoding='utf8')

meta = {
    'vocab': vocab,
    'embedding': embedding.tolist()
}
with open('SQuAD/meta.msgpack', 'wb') as f:
    msgpack.dump(meta, f)

result = {
    'trn_question_ids': question_ids[:len(train)],
    'dev_question_ids': question_ids[len(train):],
    'trn_context_ids': context_ids[:len(train)],
    'dev_context_ids': context_ids[len(train):],
    'trn_context_features': context_features[:len(train)], # exact match, tf
    'dev_context_features': context_features[len(train):],
    'trn_context_tags': context_tag_ids[:len(train)], # POS tagging
    'dev_context_tags': context_tag_ids[len(train):],
    'trn_context_ents': context_ent_ids[:len(train)], # Entity recognition
    'dev_context_ents': context_ent_ids[len(train):],
    'trn_context_text': context_text[:len(train)], # pure text
    'dev_context_text': context_text[len(train):],
    'trn_context_spans': context_token_span[:len(train)],
    'dev_context_spans': context_token_span[len(train):]
}
with open('SQuAD/data.msgpack', 'wb') as f:
    msgpack.dump(result, f)

if args.sample_size:
    sample_size = args.sample_size
    sample = {
        'trn_question_ids': result['trn_question_ids'][:sample_size],
        'dev_question_ids': result['dev_question_ids'][:sample_size],
        'trn_context_ids': result['trn_context_ids'][:sample_size],
        'dev_context_ids': result['dev_context_ids'][:sample_size],
        'trn_context_features': result['trn_context_features'][:sample_size],
        'dev_context_features': result['dev_context_features'][:sample_size],
        'trn_context_tags': result['trn_context_tags'][:sample_size],
        'dev_context_tags': result['dev_context_tags'][:sample_size],
        'trn_context_ents': result['trn_context_ents'][:sample_size],
        'dev_context_ents': result['dev_context_ents'][:sample_size],
        'trn_context_text': result['trn_context_text'][:sample_size],
        'dev_context_text': result['dev_context_text'][:sample_size],
        'trn_context_spans': result['trn_context_spans'][:sample_size],
        'dev_context_spans': result['dev_context_spans'][:sample_size]
    }
    with open('SQuAD/sample.msgpack', 'wb') as f:
        msgpack.dump(sample, f)
log.info('saved to disk.')
