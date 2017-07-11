import re
import os
import sys
import random
import string
import logging
import argparse
import unicodedata
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import json
import numpy as np
import pandas as pd

#===========================================================================
#================= All for preprocessing SQuAD data set ====================
#===========================================================================

def flatten_json(file, proc_func):
    with open(file, encoding="utf8") as f:
        data = json.load(f)['data']
    rows = []
    for i in range(len(data)):
        rows.append(proc_func(data[i]))
    rows = sum(rows, [])
    return rows

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def load_glove_vocab(file, wv_dim):
    vocab = set()
    with open(file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            vocab.add(token)
    return vocab

def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '

def pre_proc(text):
    # make hyphens, spaces clean
    text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

def feature_gen(C_docs, Q_docs, no_match):
    C_tags = [[w.tag_ for w in doc] for doc in C_docs]
    C_ents = [[w.ent_type_ for w in doc] for doc in C_docs]
    C_features = []
    for question, context in zip(Q_docs, C_docs):
        counter_ = Counter(w.text.lower() for w in context)
        total = sum(counter_.values())
        term_freq = [counter_[w.text.lower()] / total for w in context]
        if no_match:
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

def get_context_span(context, context_token):
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
            log.info("Something wrong with get_context_span()")
            return []
        t_span.append((p_str, p_str + token_len))

        p_str += token_len
        p_token += 1
    return t_span

def build_embedding(embed_file, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0 # <PAD> should be all 0 (using broadcast)

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids

#===========================================================================
#================ For batch generation (train & predict) ===================
#===========================================================================

class BatchGen:
    def __init__(self, data, batch_size, gpu, evaluation=False):
        '''
        input:
            data - list of lists
            batch_size - int
        '''
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # random shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 7
            else:
                assert len(batch) == 9

            context_len = max(len(x) for x in batch[0])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[0]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[1][0][0])
            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                context_tag[i, :len(doc)] = torch.LongTensor(doc)

            context_ent = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[3]):
                context_ent[i, :len(doc)] = torch.LongTensor(doc)

            question_len = max(len(x) for x in batch[4])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[4]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)

            if not self.eval:
                y_s = torch.LongTensor(batch[5])
                y_e = torch.LongTensor(batch[6])

            text = list(batch[-2]) # raw text
            span = list(batch[-1]) # character span for each words

            if self.gpu: # page locked memory for async data transfer
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()

            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span)

#===========================================================================
#=================== For standard evaluation in SQuAD ======================
#===========================================================================

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False

def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)

def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1
