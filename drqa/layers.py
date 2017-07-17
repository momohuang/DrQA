# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

# No modification is made to this file.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class MultiAttnMatch(nn.Module):
    """
    Given sequences X and Y, match sequence Y to each element in X through multi-attention
    """
    def __init__(self, d_in, d_key, d_val, h, do_relu = False):
        super(MultiAttnMatch, self).__init__()
        self.to_query =  nn.Linear(d_in, h * d_key)
        self.to_key = nn.Linear(d_in, h * d_key)
        self.to_value = nn.Linear(d_in, h * d_val)
        self.h = h
        self.d_key = d_key
        self.d_val = d_val
        self.do_relu = do_relu

    def forward(self, x, y, y_mask):
        """
        Input shapes:
            x = batch * len1 * d_in
            y = batch * len2 * d_in
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * (h*d_val)
        """
        queries = self.to_query(x.view(-1, x.size(2))).view(x.size(0), x.size(1), self.h, self.d_key)
        if self.do_relu:
            queries = F.relu(queries)
        keys = self.to_key(y.view(-1, y.size(2))).view(x.size(0), y.size(1), self.h, self.d_key)
        if self.do_relu:
            keys = F.relu(keys)
        scores = torch.bmm(queries.transpose(1,2).contiguous().view(-1, x.size(1), self.d_key),
            keys.transpose(1,2).transpose(2,3).contiguous().view(-1, self.d_key, y.size(1)))

        scores = scores.view(x.size(0), -1, y.size(1))
        scores = scores / (d_key ** 0.5)
        # scores: batch * (h*len1) * len2 (note the order of h, len1)

        y_mask = y_mask.unsqueeze(1).expand_as(scores)
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        alpha = F.softmax(scores.view(-1, y.size(1))).view(-1, x.size(1), y.size(1))
        # alpha: (batch*h) * len1 * len2

        values = self.to_value(y.view(-1, y.size(2))).view(x.size(0), y.size(1), self.h, self.d_val)
        values = values.transpose(1,2).contiguous().view(-1, y.size(1), self.d_val)
        if self.do_relu:
            values = F.relu(values)
        # values: (batch*h) * len2 * d_val

        matched_seq = torch.bmm(alpha, values).view(x.size(0), self.h, x.size(1), self.d_val).tranpose(1,2)
        # matched_seq: batch * len1 * h * d_val

        return matched_seq.contiguous().view(x.size(0), x.size(1), self.h * self.d_val)


class SeqAttnMatch(nn.Module):
    """
    Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_ij * y_j) for i in X
    * alpha_ij = softmax(proj(y_j) * proj(x_i))
    """
    def __init__(self, input_size, attention_type='relu_FC', normalize='sumY'):
        super(SeqAttnMatch, self).__init__()
        if attention_type == 'relu_FC':
            self.linear = nn.Linear(input_size, input_size)
        elif attention_type == 'inner_prod':
            self.linear = None
        elif attention_type == 'trainable_inner_prod':
            self.weight = Parameter(torch.Tensor(input_size))
            stdv = 1. / math.sqrt(input_size)
            self.weight.data.uniform_(-stdv, stdv)
        elif attention_type == 'trainable_inner_prod_ext':
            self.weight = Parameter(torch.Tensor(input_size))
            stdv = 1. / math.sqrt(input_size)
            self.weight.data.uniform_(-stdv, stdv)
            self.ulinear = nn.Linear(input_size, 1, bias = False)
            self.vlinear = nn.Linear(input_size, 1, bias = False)
        elif attention_type == 'MLP':
            self.ulinear = nn.Linear(input_size, input_size)
            self.vlinear = nn.Linear(input_size, input_size, bias = False)
            self.tlinear = nn.Linear(input_size, 1, bias = False)
        else:
            raise NotImplementedError('attention_type = %s' % attention_type)
        self.attention_type = attention_type
        self.input_size = input_size

        if normalize != 'sumX' and normalize != 'sumY':
            raise NotImplementedError('normalize = %s' % normalize)
        self.normalize = normalize

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.attention_type == 'relu_FC':
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
            scores = x_proj.bmm(y_proj.transpose(2, 1))
        elif self.attention_type == 'inner_prod':
            scores = x.bmm(y.transpose(2, 1))
        elif self.attention_type == 'trainable_inner_prod':
            y_weight = y * self.weight.unsqueeze(0).unsqueeze(1).expand_as(y)
            scores = x.bmm(y_weight.transpose(2, 1))
        elif self.attention_type == 'trainable_inner_prod_ext':
            y_weight = y * self.weight.unsqueeze(0).unsqueeze(1).expand_as(y)
            scores = x.bmm(y_weight.transpose(2, 1))
            x_bias = self.ulinear(x.view(-1, x.size(2))).view(x.size(0), x.size(1), 1).expand_as(scores)
            y_bias = self.vlinear(y.view(-1, y.size(2))).view(y.size(0), 1, y.size(1)).expand_as(scores)
            scores = scores + x_bias + y_bias
        elif self.attention_type == 'MLP':
            x_proj = self.ulinear(x.view(-1, x.size(2))).view(x.size()).unsqueeze(2).expand(x.size(0), x.size(1), y.size(1), x.size(2))
            y_proj = self.vlinear(y.view(-1, y.size(2))).view(y.size()).unsqueeze(1).expand(x.size(0), x.size(1), y.size(1), x.size(2))
            scores = self.tlinear(F.tanh(x_proj + y_proj).view(-1, x.size(2))).view(x.size(0), x.size(1), y.size(1))

        # scores = batch * len1 * len2

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        if self.normalize == 'sumY':
            alpha_flat = F.softmax(scores.view(-1, y.size(1)))
            alpha = alpha_flat.view(-1, x.size(1), y.size(1))
        if self.normalize == 'sumX':
            alpha_flat = F.softmax(scores.transpose(1,2).contiguous().view(-1, x.size(1)))
            alpha = alpha_flat.view(-1, y.size(1), x.size(1)).transpose(1,2)

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy

class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha

class ChoiceLayer(nn.Module):
    def __init__(self, input_size):
        super(ChoiceLayer, self).__init__()
        self.linear = nn.Linear(2 * input_size, input_size)
    def forward(self, x1, x2):
        """
        xi = batch * len * hdim
        """
        assert(x1.size() == x2.size())
        g = F.sigmoid(self.linear(torch.cat((x1, x2), 2).view(-1, 2*x1.size(-1))).view(x1.size()))
        return g * x1 + (1-g) * x2

class GatedLayer(nn.Module):
    def __init__(self, input_size):
        super(GatedLayer, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
    def forward(self, x):
        """
        x = batch * len * hdim
        """
        x_flat = x.view(-1, x.size(-1))
        g = F.sigmoid(self.linear(x_flat).view(x.size()))
        return g * x


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------

# by default in PyTorch, +-*/ are all element-wise
def uniform_weights(x, x_mask): # used in rnn_reader.py
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha

# bmm: batch matrix multiplication
# unsqueeze: add singleton dimension
# squeeze: remove singleton dimension
def weighted_avg(x, weights): # used in rnn_reader.py
    """ x = batch * len * d
        weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)
