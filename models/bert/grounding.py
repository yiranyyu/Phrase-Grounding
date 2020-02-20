from typing import Dict, Any, Tuple, Callable, Union

import torch
import math
import torch.nn as nn
from models.base.modal_fusion import MLBFusion, MutanFusion
from util import logging


def assert_shape(tensor, expected):
    assert tensor.shape == expected, f'Expected shape {expected}, got {tensor.shape}'


def broadcast_to_match(encT, encI, n_tok, n_RoI, B, T_hidden):
    encT = encT.view(B, n_tok, 1, T_hidden)
    encT = encT.repeat(1, 1, n_RoI, 1)
    encT = encT.view(B, -1, T_hidden)
    encI = encI.repeat(1, n_tok, 1)
    return encT, encI


class AbstractGrounding(nn.Module):
    def __init__(self, cfgT, cfgI, heads=1):
        super(AbstractGrounding, self).__init__()
        self.cfgT = cfgT
        self.cfgI = cfgI

        self.num_attention_heads = heads
        self.projection = cfgI.hidden_size // 2
        self.attention_head_size = int(self.projection // self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.hidden_size = self.projection
        self.text_hidden_size = cfgT.hidden_size
        self.imag_hidden_size = cfgI.hidden_size
        logging.info(f'Grounding using {self.__class__.__name__}')


class CosineGrounding(AbstractGrounding):
    def __init__(self, cfgT, cfgI, heads=1):
        super(CosineGrounding, self).__init__(cfgT, cfgI, heads)
        projection = cfgI.hidden_size // 2
        self.attention_head_size = int(projection // self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.Q = nn.Linear(cfgT.hidden_size, self.all_head_size)
        self.K = nn.Linear(cfgI.hidden_size, self.all_head_size)

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # B x #tokens x #heads x head_size
        return x.permute(0, 2, 1, 3)  # B x #heads x # tokens x head_size

    def forward(self, encT, encI, mask):
        Q = self.Q(encT)
        K = self.K(encI)
        Q = self.transpose(Q)
        K = self.transpose(K)

        logits = torch.matmul(Q, K.transpose(-1, -2))
        logits = logits / math.sqrt(self.attention_head_size)
        logits = logits + mask
        return logits.squeeze()
        # scores = nn.Sigmoid(logits)
        # return scores


class LinearSumGrounding(AbstractGrounding):
    def __init__(self, cfgT, cfgI):
        super(LinearSumGrounding, self).__init__(cfgT, cfgI)

        self.Q_mlp = nn.Sequential(
            nn.Linear(cfgT.hidden_size, self.projection))
        self.K_mlp = nn.Sequential(
            nn.Linear(cfgI.hidden_size, self.projection))
        self.mlp = nn.Sequential(
            nn.Linear(self.projection, 1))

    def forward(self, encT: torch.Tensor, encI: torch.Tensor, mask: torch.Tensor):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        encT = self.Q_mlp(encT)  # (B, n_tok, H)
        encI = self.K_mlp(encI)  # (B, n_RoI, H)

        encT, encI = broadcast_to_match(encT, encI, n_tok, n_RoI, B, self.hidden_size)

        fusion = torch.tanh(encI + encT)
        logits = self.mlp(fusion)
        logits = logits.view(B, n_tok, n_RoI)
        logits = logits + mask.squeeze(1)
        return logits


class LinearConcatenateGrounding(AbstractGrounding):
    def __init__(self, cfgT, cfgI):
        super(LinearConcatenateGrounding, self).__init__(cfgT, cfgI)

        self.Q_mlp = nn.Sequential(
            nn.Linear(cfgT.hidden_size, self.projection),
            nn.ReLU())
        self.K_mlp = nn.Sequential(
            nn.Linear(cfgI.hidden_size, self.projection),
            nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(self.projection * 2, self.projection),
            nn.ReLU(),
            nn.Linear(self.projection, 1))

    def forward(self, encT: torch.Tensor, encI: torch.Tensor, mask: torch.Tensor):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        encT = self.Q_mlp(encT)  # (B, n_tok, H)
        encI = self.K_mlp(encI)  # (B, n_RoI, H)

        encT, encI = broadcast_to_match(encT, encI, n_tok, n_RoI, B, self.hidden_size)

        fusion = torch.cat([encI, encT], dim=2)
        logits = self.mlp(fusion)
        logits = logits.view(B, n_tok, n_RoI)
        logits = logits + mask.squeeze(1)
        return logits


class MutanGrounding(AbstractGrounding):
    def __init__(self, cfgT, cfgI):
        super(MutanGrounding, self).__init__(cfgT, cfgI)

        self.mm_hidden = 510
        self.fusion = MutanFusion(self.text_hidden_size, self.imag_hidden_size, self.mm_hidden)
        self.score = nn.Linear(self.mm_hidden, 1)

    def forward(self, encT, encI, mask):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        encT, encI = broadcast_to_match(encT, encI, n_tok, n_RoI, B, T_hidden)
        fusion = self.fusion(encT, encI)
        logits = self.score(fusion).view(B, n_tok, n_RoI)
        logits = logits + mask.squeeze(1)
        return logits


class MLBGrounding(AbstractGrounding):
    def __init__(self, cfgT, cfgI):
        super(MLBGrounding, self).__init__(cfgT, cfgI)

        self.mm_hidden = 1200
        self.fusion = MLBFusion(self.text_hidden_size, self.imag_hidden_size, self.mm_hidden)
        self.score = nn.Linear(self.mm_hidden, 1)

    def forward(self, encT, encI, mask):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        encT, encI = broadcast_to_match(encT, encI, n_tok, n_RoI, B, T_hidden)
        fusion = self.fusion(encT, encI)
        logits = self.score(fusion).view(B, n_tok, n_RoI)
        logits = logits + mask.squeeze(1)
        return logits


class FusionFusionGrounding(AbstractGrounding):
    # To suppress IDE warning
    att_fusion: Union[Tuple[Callable[[Tuple[Any, ...], Dict[str, Any]], Any]], Callable]
    cls_fusion: Union[Tuple[Callable[[Tuple[Any, ...], Dict[str, Any]], Any]], Callable]

    def __init__(self,
                 cfgT,
                 cfgI,
                 attention_fusion=MutanGrounding,
                 classification_fusion=MutanGrounding):
        super(FusionFusionGrounding, self).__init__(cfgT, cfgI)

        self.attention = attention_fusion(cfgT, cfgI)

        cfgT.hidden_size = self.text_hidden_size + self.imag_hidden_size
        cfgI.hidden_size = self.text_hidden_size + self.imag_hidden_size
        self.classification = classification_fusion(cfgT, cfgI)

    def forward(self, encT, encI, mask):
        # (B, n_tok, n_RoI)
        attention = self.attention(encT, encI, mask)
        attention_on_T: torch.Tensor = attention.softmax(dim=1)
        attention_on_I: torch.Tensor = attention.softmax(dim=2)

        attented_T = torch.bmm(attention_on_T.permute(0, 2, 1), encT)
        attented_I = torch.bmm(attention_on_I, encI)
        fused_T = torch.cat([encT, attented_I], dim=-1)
        fused_I = torch.cat([encI, attented_T], dim=-1)

        logits = self.classification(fused_T, fused_I, mask)
        logits = logits + attention + mask.squeeze(1)
        return logits
