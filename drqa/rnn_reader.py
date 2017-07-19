# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers

# Modification: add 'pos' and 'ner' features.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, embedding=None, padding_idx=0):
        super(RnnDocReader, self).__init__()
        # Word embeddings
        self.embedding = nn.Embedding(opt['vocab_size'],
                                      opt['embedding_dim'],
                                      padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data = embedding
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial'] + 2:]
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

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=opt['embedding_dim'],
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Inter-alignment
        int_ali_doc_hidden_size = doc_hidden_size
        int_ali_question_hidden_size = question_hidden_size

        if opt['do_C2Q']:
            assert(doc_hidden_size == question_hidden_size)
            if opt['do_multi_att']:
                self.inter_align = layers.MultiAttnMatch(doc_hidden_size, opt['multi_att_key'], opt['multi_att_val'], opt['multi_att_h'], do_relu = opt['multi_att_do_relu'], att_dropout_p = opt['multi_att_dropout'])

                # currently always do concat, because the size may differ
                if opt['inter_att_concat'] != 'concat':
                    print('\"inter_att_concat\" option only supports [concat] (changed to [concat])')
                    opt['inter_att_concat'] = 'concat'
                int_ali_doc_hidden_size += opt['multi_att_h'] * opt['multi_att_val']
            else:
                self.inter_align = layers.SeqAttnMatch(doc_hidden_size, opt['inter_att_type'])

                if opt['inter_att_concat'] == 'fuse':
                    int_ali_doc_hidden_size *= 1
                    self.fusion = layers.ChoiceLayer(doc_hidden_size)
                elif opt['inter_att_concat'] == 'concat':
                    int_ali_doc_hidden_size *= 2
                elif opt['inter_att_concat'] == 'concat_dot':
                    int_ali_doc_hidden_size *= 3
                elif opt['inter_att_concat'] == 'concat_dot_diff':
                    int_ali_doc_hidden_size *= 4
                else:
                    raise NotImplementedError('inter_att_concat: %s' % opt['inter_att_concat'])

        if opt['do_my_Q2C']:
            int_ali_question_hidden_size += doc_hidden_size
            if opt['do_multi_att']:
                self.my_Q2C = layers.MultiAttnMatch(doc_hidden_size, opt['multi_att_key'], opt['multi_att_val'], opt['multi_att_h'], do_relu = opt['multi_att_do_relu'], att_dropout_p = opt['multi_att_dropout'])
            else:
                self.my_Q2C = layers.SeqAttnMatch(doc_hidden_size, opt['inter_att_type'])

        if opt['do_coattention']:
            assert(doc_hidden_size == question_hidden_size)

            int_ali_doc_hidden_size += doc_hidden_size
            if opt['do_multi_att']:
                assert(doc_hidden_size == opt['multi_att_h'] * opt['multi_att_val'])

                self.context4query = layers.MultiAttnMatch(doc_hidden_size, opt['multi_att_key'], opt['multi_att_val'], opt['multi_att_h'], do_relu = opt['multi_att_do_relu'], att_dropout_p = opt['multi_att_dropout'])
                self.coattention = layers.MultiAttnMatch(doc_hidden_size, opt['multi_att_key'], opt['multi_att_val'], opt['multi_att_h'], do_relu = opt['multi_att_do_relu'], att_dropout_p = opt['multi_att_dropout'])
            else:
                self.context4query = layers.SeqAttnMatch(doc_hidden_size, opt['inter_att_type'])
                self.coattention = layers.SeqAttnMatch(doc_hidden_size, opt['inter_att_type'])

        if opt['do_C2Q'] or opt['do_coattention']:
            print('Inter-aligned doc vector size = ', int_ali_doc_hidden_size)

            # Gated layer
            if opt['gated_int_ali_doc']:
                self.int_ali_doc_gate = layers.GatedLayer(input_size=int_ali_doc_hidden_size)

            # Constructing LSTM after inter-alignment
            if opt['int_ali_hidden_size'] != -1:
                hsize = opt['int_ali_hidden_size']
            else:
                hsize = opt['hidden_size']

            doc_final_hidden_size = 2 * hsize
            if opt['concat_rnn_layers']:
                doc_final_hidden_size *= opt['doc_layers']

            self.inter_align_rnn = layers.StackedBRNN(
                input_size=int_ali_doc_hidden_size,
                hidden_size=hsize,
                num_layers=opt['doc_layers'],
                dropout_rate=opt['dropout_rnn'],
                dropout_output=opt['dropout_rnn_output'],
                concat_layers=opt['concat_rnn_layers'],
                rnn_type=self.RNN_TYPES[opt['rnn_type']],
                padding=opt['rnn_padding'],
            )
        else:
            doc_final_hidden_size = doc_hidden_size

        if opt['do_my_Q2C']:
            print('Inter-aligned question vector size = ', int_ali_question_hidden_size)

            # Gated layer
            if opt['gated_int_ali_question']:
                self.int_ali_question_gate = layers.GatedLayer(input_size=int_ali_question_hidden_size)

            # Constructing LSTM after inter-alignment
            if opt['int_ali_hidden_size'] != -1:
                hsize = opt['int_ali_hidden_size']
            else:
                hsize = opt['hidden_size']

            question_final_hidden_size = 2 * hsize
            if opt['concat_rnn_layers']:
                question_final_hidden_size *= opt['question_layers']

            self.inter_align_question_rnn = layers.StackedBRNN(
                input_size=int_ali_question_hidden_size,
                hidden_size=hsize,
                num_layers=opt['question_layers'],
                dropout_rate=opt['dropout_rnn'],
                dropout_output=opt['dropout_rnn_output'],
                concat_layers=opt['concat_rnn_layers'],
                rnn_type=self.RNN_TYPES[opt['rnn_type']],
                padding=opt['rnn_padding'],
            )
        else:
            question_final_hidden_size = question_hidden_size

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_final_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_final_hidden_size,
            question_final_hidden_size,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_final_hidden_size,
            question_final_hidden_size,
        )

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
        if self.training:
            x1_emb = self.embedding(x1)
            x2_emb = self.embedding(x2)
        else:
            x1_emb = self.eval_embed(x1)
            x2_emb = self.eval_embed(x2)

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
        if self.opt['do_C2Q']:
            C2Q_hiddens = self.inter_align(doc_hiddens, question_hiddens, x2_mask)
            if self.opt['inter_att_concat'] == 'fuse':
                doc_int_ali_input = self.fusion(doc_hiddens, C2Q_hiddens)
            elif self.opt['inter_att_concat'] == 'concat':
                doc_int_ali_input = torch.cat((doc_hiddens, C2Q_hiddens), 2)
            elif self.opt['inter_att_concat'] == 'concat_dot':
                doc_int_ali_input = torch.cat((doc_hiddens, C2Q_hiddens, doc_hiddens*C2Q_hiddens), 2)
            elif self.opt['inter_att_concat'] == 'concat_dot_diff':
                doc_int_ali_input = torch.cat((doc_hiddens, C2Q_hiddens, doc_hiddens*C2Q_hiddens, doc_hiddens-C2Q_hiddens), 2)

        if self.opt['do_coattention']:
            C4Q_hiddens = self.context4query(question_hiddens, doc_hiddens, x1_mask)
            coatt_hiddens = self.coattention(doc_hiddens, C4Q_hiddens, x2_mask)
            doc_int_ali_input = torch.cat((doc_int_ali_input, coatt_hiddens), 2)

        if self.opt['do_my_Q2C']:
            my_Q2C_hiddens = self.my_Q2C(question_hiddens, doc_hiddens, x1_mask)
            question_int_ali_input = torch.cat((question_hiddens, my_Q2C_hiddens), 2)

        # LSTM after inter-alignment
        if self.opt['do_C2Q'] or self.opt['do_coattention']:
            if self.opt['gated_int_ali_doc']:
                doc_int_ali_input = self.int_ali_doc_gate(doc_int_ali_input)
            doc_final_hiddens = self.inter_align_rnn(doc_int_ali_input, x1_mask)
        else:
            doc_final_hiddens = doc_hiddens

        if self.opt['do_my_Q2C']:
            if self.opt['gated_int_ali_question']:
                question_int_ali_input = self.int_ali_question_gate(question_int_ali_input)
            question_final_hiddens = self.inter_align_question_rnn(question_int_ali_input, x2_mask)
        else:
            question_final_hiddens = question_hiddens

        # Merge question hiddens for answer generation
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_final_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_final_hiddens, x2_mask)
        question_final_hidden = layers.weighted_avg(question_final_hiddens, q_merge_weights)

        # Predict scores for starting and ending position
        start_scores = self.start_attn(doc_final_hiddens, question_final_hidden, x1_mask)
        end_scores = self.end_attn(doc_final_hiddens, question_final_hidden, x1_mask)
        return start_scores, end_scores # -inf to inf
