# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers

class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    def __init__(self, opt, embedding=None, padding_idx=0):
        super(RnnDocReader, self).__init__()
        # Word embeddings
        self.embedding = nn.Embedding(opt['vocab_size'],
                                      opt['embedding_dim'],
                                      padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data = embedding
            if opt['fix_embeddings'] or opt['tune_partial'] == 0:
                opt['fix_embeddings'] = True
                opt['tune_partial'] = 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            else:
                assert opt['tune_partial'] < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial']:]
                # a persistent buffer for the nn.Module
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding

        if opt['pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
        if opt['ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
        # Projection for attention weighted question
        if opt['wvec_align']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['wvec_align']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_dim']
        if opt['ner']:
            doc_input_size += opt['ner_dim']

        # Gated layer
        if opt['gated_input']:
            self.gated_input = layers.GatedLayer(input_size=doc_input_size)

        # Setup the vector size for [doc, question]
        # they will be modified in the following code
        doc_hidden_size, que_hidden_size = doc_input_size, opt['embedding_dim']
        print('Initially, the vector_sizes [doc, query] are', doc_hidden_size, que_hidden_size)

        # RNN document encoder
        self.doc_rnn, doc_hidden_size = layers.RNN_from_opt(doc_hidden_size, opt['hidden_size'], opt)

        # RNN question encoder
        self.question_rnn, que_hidden_size = layers.RNN_from_opt(que_hidden_size, opt['hidden_size'], opt)

        # Output sizes of rnn encoders
        print('After LSTM, the vector_sizes [doc, query] are', doc_hidden_size, que_hidden_size)

        # Inter-alignment
        if opt['do_coattention'] and opt['do_C2Q']:
            print('Doing coattention also does C2Q, turning off C2Q')
            opt['do_C2Q'] = False

        if opt['do_C2Q']:
            self.C2Q = layers.Unidir_atten(opt, doc_hidden_size, que_hidden_size)
        if opt['do_coattention']:
            self.coattention = layers.Coattention(opt, doc_hidden_size, que_hidden_size)
        if opt['do_my_Q2C']:
            self.my_Q2C = layers.Unidir_atten(opt, doc_hidden_size, que_hidden_size, reverse=True)

        doc_hidden_size = self.coattention.output_size if opt['do_coattention'] else \
                          self.C2Q.output_size         if opt['do_C2Q'] else \
                          doc_hidden_size
        que_hidden_size = self.my_Q2C.output_size      if opt['do_my_Q2C'] else \
                          que_hidden_size
        print('After inter-alignment, the vector_sizes [doc, query] are', doc_hidden_size, que_hidden_size)

        # Contextual integration
        if opt['do_C2Q'] or opt['do_coattention']:
            # Gated layer
            if opt['gated_int_ali_doc']:
                self.int_ali_doc_gate = layers.GatedLayer(input_size=vector_sizes[0])

            # Constructing LSTM after inter-alignment
            self.int_ali_doc_rnn, doc_hidden_size = layers.RNN_from_opt(doc_hidden_size,
            opt['int_ali_hidden_size'] if opt['int_ali_hidden_size'] != -1 else opt['hidden_size'], opt)

        if opt['do_my_Q2C']:
            # Gated layer
            if opt['gated_int_ali_question']:
                self.int_ali_question_gate = layers.GatedLayer(input_size=vector_sizes[1])

            # Constructing LSTM after inter-alignment
            self.int_ali_question_rnn, que_hidden_size = layers.RNN_from_opt(que_hidden_size,
            opt['int_ali_hidden_size'] if opt['int_ali_hidden_size'] != -1 else opt['hidden_size'], opt)

        print('After LSTM, the vector_sizes [doc, query] are', doc_hidden_size, que_hidden_size)

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(que_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(doc_hidden_size, que_hidden_size)
        self.end_attn = layers.BilinearSeqAttn(doc_hidden_size, que_hidden_size)

        # Store config
        self.opt = opt

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Word embedding for both document and question
        emb = self.embedding if self.training else self.eval_embed
        x1_emb, x2_emb = emb(x1), emb(x2)

        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                           training=self.training)

        drnn_input_list = [x1_emb, x1_f]
        # Add attention-weighted question representation
        if self.opt['wvec_align']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            drnn_input_list.append(x1_pos_emb)
        if self.opt['ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            drnn_input_list.append(x1_ner_emb)

        if self.opt['gated_input']:
            x1_input = self.gated_input(torch.cat(drnn_input_list, 2))
        else:
            x1_input = torch.cat(drnn_input_list, 2)
        x2_input = x2_emb

        # Now the features are ready
        # x1_input: [batch_size, doc_len, x1_feats]
        #           including wvec, match, tf, Qemb, pos, ner
        # x2_input: [batch_size, doc_len, x2_feats]
        #           including wvec

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(x1_input, x1_mask)

        # Encode question with RNN
        question_hiddens = self.question_rnn(x2_input, x2_mask)

        # Inter-alignment
        doc_align = self.coattention if self.opt['do_coattention'] else \
                    self.C2Q         if self.opt['do_C2Q'] else \
                    lambda a,b,c,d: a
        que_align = self.my_Q2C      if self.opt['do_my_Q2C'] else \
                    lambda a,b,c,d: b
        doc_hiddens, question_hiddens = doc_align(doc_hiddens, question_hiddens, x1_mask, x2_mask), \
                                        que_align(doc_hiddens, question_hiddens, x1_mask, x2_mask)

        # RNN after inter-alignment
        if self.opt['do_C2Q'] or self.opt['do_coattention']:
            if self.opt['gated_int_ali_doc']:
                doc_hiddens = self.int_ali_doc_gate(doc_hiddens)
            doc_hiddens = self.int_ali_doc_rnn(doc_hiddens, x1_mask)

        if self.opt['do_my_Q2C']:
            if self.opt['gated_int_ali_question']:
                question_hiddens = self.int_ali_question_gate(question_hiddens)
            question_hiddens = self.int_ali_question_rnn(question_hiddens, x2_mask)

        # Merge question hiddens for answer generation
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hiddens = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict scores for starting and ending position
        start_scores = self.start_attn(doc_hiddens, question_hiddens, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hiddens, x1_mask)
        return start_scores, end_scores # -inf to inf
