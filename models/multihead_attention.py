import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from softmax_one.softmax_one import softmax_one

__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention','ScaledDotProductAttentionCustom']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, temp,mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        #if mask is not None:()
        #scores = scores.masked_fill(mask == 0, -1e9)
        #attention = F.softmax(scores, dim=-1)
        id = torch.diagonal(scores,0,2)

        #for 9 channel 5
        #for HHAR 3 (might revise again)
        #1 for sim
        #for Simulations (10 for wisdm)?

        #5 for temp
        id = F.softmax(id/temp
                       ,dim=-1)
        #id = torch.nn.ReLU(id, dim=-1)
        attention = torch.diag_embed(id)
        #attention = scores*torch.eye(3,3).cuda()
        #attention = F.softmax(scores, dim=-1)
        return attention.matmul(value),attention



'''
class ScaledDotProductAttentionFull(nn.Module):

    def forward(self, query, key, value, temp=1,mask=None):
        dk = query.size()[-1]
        scores = torch.abs(query.matmul(key.transpose(-2, -1)) / math.sqrt(dk))
        #if mask is not None:()
        #scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores/1e-2, dim=-1)
        #attention = F.normalize(scores,dim=-1)

        return attention.matmul(value),attention

'''


class ScaledDotProductAttentionCustom(nn.Module):

    def forward(self, query, key, value, temp=1,mask=None):

        dk = query.size()[-1]
        scores = torch.abs(query.matmul(key.transpose(-2, -1)) / math.sqrt(dk))
        id = torch.diagonal(scores, 0, 2)
        id = F.softmax(id /temp
                       , dim=-1)
        attention_mask = torch.diag_embed(id)

        attention_diag = F.softmax(scores, dim=-1)
        torch.diag_embed(id)
        attention = id.unsqueeze(2)*attention_diag
        attention = id.unsqueeze(1)*attention
        #attention = F.normalize(scores,dim=-1)

        return attention.matmul(value),attention

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,temp,
                 bias=True,
                 activation=None):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.temp = temp
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, temp=1, mask=None):
        #q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        #modify to get no transofrmation on v
        q, k, v = self.linear_q(q), self.linear_k(k), v
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y,attention = ScaledDotProductAttention()(q, k, v, self.temp, mask)
        #y, attention = ScaledDotProductAttentionCustom()(q, k, v, temp, mask)
        y = self._reshape_from_batches(y)

        #y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y,attention

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )