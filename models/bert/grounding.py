import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def assert_shape(tensor, expected):
    assert tensor.shape == expected, f'Expected shape {expected}, got {tensor.shape}'


class CosineGrounding(nn.Module):
    def __init__(self, cfgT, cfgI, heads=1):
        super(CosineGrounding, self).__init__()
        projection = cfgI.hidden_size // 2
        self.num_attention_heads = heads
        self.attention_head_size = int(projection // self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.Q = nn.Linear(cfgT.hidden_size, self.all_head_size)
        self.K = nn.Linear(cfgI.hidden_size, self.all_head_size)
        self.cfgT = cfgT
        self.cfgI = cfgI
        # self.dropout = nn.Dropout(cfgI.attention_probs_dropout_prob)

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


# NOTE: OOM
class BilinearGrounding(nn.Module):
    def __init__(self, cfgT, cfgI):
        super(BilinearGrounding, self).__init__()
        projection = cfgI.hidden_size // 2
        self.hidden_size = projection
        self.text_hidden_size = cfgT.hidden_size
        self.imag_hidden_size = cfgI.hidden_size

        self.bilinear = nn.Bilinear(self.text_hidden_size, self.imag_hidden_size, 1)

    def forward(self, encT: torch.Tensor, encI: torch.Tensor, mask: torch.Tensor):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        encT = encT.view(B, n_tok, 1, T_hidden)
        encT = encT.repeat(1, 1, n_RoI, 1)
        encT = encT.view(B, -1, T_hidden)
        encI = encI.repeat(1, n_tok, 1)

        logits = self.bilinear(encT, encI).view(B, n_tok, n_RoI)
        logits = logits + mask.squeeze(1)
        return logits


class MUTANGrounding(nn.Module):
    pass


class LinearSumGrounding(nn.Module):
    def __init__(self, cfgT, cfgI):
        super(LinearSumGrounding, self).__init__()
        projection = cfgI.hidden_size // 2
        self.hidden_size = projection
        self.text_hidden_size = cfgT.hidden_size
        self.imag_hidden_size = cfgI.hidden_size

        # TODO: test result without bias
        self.Q_mlp = nn.Sequential(
            nn.Linear(cfgT.hidden_size, projection),
        )
        self.K_mlp = nn.Sequential(
            nn.Linear(cfgI.hidden_size, projection),
        )
        self.mlp = nn.Sequential(
            nn.Linear(projection, 1)
        )
        print('Linear Sum', flush=True)

    def forward(self, encT: torch.Tensor, encI: torch.Tensor, mask: torch.Tensor):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        encT = self.Q_mlp(encT)  # (B, n_tok, H)
        encI = self.K_mlp(encI)  # (B, n_RoI, H)

        encT = encT.view(B, n_tok, 1, self.hidden_size)
        encT = encT.repeat(1, 1, n_RoI, 1)
        encT = encT.view(B, -1, self.hidden_size)
        encI = encI.repeat(1, n_tok, 1)

        fusion = torch.tanh(encI + encT)
        logits = self.mlp(fusion)
        logits = logits.view(B, n_tok, n_RoI)
        logits = logits + mask.squeeze(1)
        return logits


class LinearConcatenateGrounding(nn.Module):
    def __init__(self, cfgT, cfgI):
        super(LinearConcatenateGrounding, self).__init__()
        projection = cfgI.hidden_size // 2
        self.hidden_size = projection
        self.text_hidden_size = cfgT.hidden_size
        self.imag_hidden_size = cfgI.hidden_size

        self.Q_mlp = nn.Sequential(
            nn.Linear(cfgT.hidden_size, projection),
            nn.ReLU()
        )
        self.K_mlp = nn.Sequential(
            nn.Linear(cfgI.hidden_size, projection),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(projection * 2, projection),
            nn.ReLU(),
            nn.Linear(projection, 1)
        )
        print('Double Linear', flush=True)

    def forward(self, encT: torch.Tensor, encI: torch.Tensor, mask: torch.Tensor):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        encT = self.Q_mlp(encT)  # (B, n_tok, H)
        encI = self.K_mlp(encI)  # (B, n_RoI, H)

        encT = encT.view(B, n_tok, 1, self.hidden_size)
        encT = encT.repeat(1, 1, n_RoI, 1)
        encT = encT.view(B, -1, self.hidden_size)
        encI = encI.repeat(1, n_tok, 1)

        fusion = torch.cat([encI, encT], dim=2)
        logits = self.mlp(fusion)
        logits = logits.view(B, n_tok, n_RoI)
        logits = logits + mask.squeeze(1)
        return logits


class ModalFusionGrounding(nn.Module):
    def __init__(self, cfgT, cfgI, heads=1):
        super(ModalFusionGrounding, self).__init__()
        projection = cfgI.hidden_size // 2
        self.num_attention_heads = heads
        self.attention_head_size = int(projection // self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.attention = CosineGrounding(cfgT=cfgT, cfgI=cfgI)
        self.Q = nn.Linear(cfgI.hidden_size + cfgT.hidden_size, self.all_head_size)
        self.K = nn.Linear(cfgI.hidden_size + cfgT.hidden_size, self.all_head_size)

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # B x #tokens x #heads x head_size
        return x.permute(0, 2, 1, 3)  # B x #heads x # tokens x head_size

    # noinspection DuplicatedCode
    def forward(self, encT: torch.Tensor, encI: torch.Tensor, mask: torch.Tensor):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        # (B, n_tok, n_RoI)
        attention = self.attention(encT, encI, mask)
        attention_on_T: torch.Tensor = attention.softmax(dim=1)
        attention_on_I: torch.Tensor = attention.softmax(dim=2)

        attented_T = torch.bmm(attention_on_T.permute(0, 2, 1), encT)
        attented_I = torch.bmm(attention_on_I, encI)

        fused_T = torch.cat([encT, attented_I], dim=-1)
        fused_I = torch.cat([encI, attented_T], dim=-1)

        Q = self.Q(fused_T)
        K = self.K(fused_I)
        Q = self.transpose(Q)
        K = self.transpose(K)

        K = K.transpose(-1, -2)
        assert_shape(K, (B, self.num_attention_heads, self.attention_head_size, n_RoI))

        logits = torch.matmul(Q, K)
        logits = logits / math.sqrt(self.attention_head_size)
        logits = logits.squeeze() + attention + mask.squeeze(1)
        return logits.squeeze()


class ModalFusionConcatenateGrounding(nn.Module):
    def __init__(self, cfgT, cfgI, heads=1):
        super(ModalFusionConcatenateGrounding, self).__init__()
        projection = cfgI.hidden_size // 2
        self.text_hidden_size = cfgT.hidden_size
        self.imag_hidden_size = cfgI.hidden_size
        self.num_attention_heads = heads
        self.attention_head_size = int(projection // self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.attention = CosineGrounding(cfgT=cfgT, cfgI=cfgI)
        self.Q = nn.Linear(cfgI.hidden_size + cfgT.hidden_size, self.all_head_size)
        self.K = nn.Linear(cfgI.hidden_size + cfgT.hidden_size, self.all_head_size)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.all_head_size, projection),
            nn.ReLU(),
            nn.Linear(projection, 1)
        )

    # noinspection DuplicatedCode
    def forward(self, encT: torch.Tensor, encI: torch.Tensor, mask: torch.Tensor):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        # (B, n_tok, n_RoI)
        attention = self.attention(encT, encI, mask)
        attention_on_T: torch.Tensor = attention.softmax(dim=1)
        attention_on_I: torch.Tensor = attention.softmax(dim=2)

        attented_T = torch.bmm(attention_on_T.permute(0, 2, 1), encT)
        attented_I = torch.bmm(attention_on_I, encI)

        fused_T = torch.cat([encT, attented_I], dim=-1)
        fused_I = torch.cat([encI, attented_T], dim=-1)

        fused_T = self.Q(fused_T)
        fused_I = self.K(fused_I)

        fused_T = fused_T.view(B, n_tok, 1, self.all_head_size)
        fused_T = fused_T.repeat(1, 1, n_RoI, 1)
        fused_T = fused_T.view(B, -1, self.all_head_size)
        fused_I = fused_I.repeat(1, n_tok, 1)

        fusion = torch.cat([fused_I, fused_T], dim=2)
        logits = self.mlp(fusion)
        logits = logits.view(B, n_tok, n_RoI)
        logits = logits.squeeze() + attention + mask.squeeze(1)
        return logits


# NOTE: OOM, cannot run since backward costs too much memory
class ModalFusionBilinearGrounding(nn.Module):
    def __init__(self, cfgT, cfgI, heads=1):
        super(ModalFusionBilinearGrounding, self).__init__()
        projection = cfgI.hidden_size // 2
        self.text_hidden_size = cfgT.hidden_size
        self.imag_hidden_size = cfgI.hidden_size
        self.num_attention_heads = heads
        self.attention_head_size = int(projection // self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.attention = CosineGrounding(cfgT=cfgT, cfgI=cfgI)
        self.bilinear = nn.Bilinear(cfgI.hidden_size + cfgT.hidden_size, cfgI.hidden_size + cfgT.hidden_size, 1)

    def forward(self, encT: torch.Tensor, encI: torch.Tensor, mask: torch.Tensor):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        # (B, n_tok, n_RoI)
        attention = self.attention(encT, encI, mask)
        attention_on_T: torch.Tensor = attention.softmax(dim=1)
        attention_on_I: torch.Tensor = attention.softmax(dim=2)

        attented_T = torch.bmm(attention_on_T.permute(0, 2, 1), encT)
        attented_I = torch.bmm(attention_on_I, encI)

        fused_T = torch.cat([encT, attented_I], dim=-1)
        fused_I = torch.cat([encI, attented_T], dim=-1)

        fused_T = fused_T.view(B, n_tok, 1, I_hidden + T_hidden)
        fused_T = fused_T.repeat(1, 1, n_RoI, 1)
        fused_T = fused_T.view(B, -1, I_hidden + T_hidden)
        fused_I = fused_I.repeat(1, n_tok, 1)

        logits = self.bilinear(fused_T, fused_I)
        logits = logits.view(B, n_tok, n_RoI)
        logits = logits + mask.squeeze(1)
        return logits


class AbstractFusion(nn.Module):

    def __init__(self, opt={}):
        super(AbstractFusion, self).__init__()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError


class MLBFusion(AbstractFusion):

    def __init__(self, opt):
        super(MLBFusion, self).__init__(opt)
        # Modules
        if 'dim_v' in self.opt:
            self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_h'])
        else:
            print('Warning fusion.py: no visual embedding before fusion')

        if 'dim_q' in self.opt:
            self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_h'])
        else:
            print('Warning fusion.py: no question embedding before fusion')

    def forward(self, input_v, input_q):
        # visual (cnn features)
        if 'dim_v' in self.opt:
            x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
            x_v = self.linear_v(x_v)
            if 'activation_v' in self.opt:
                x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            x_v = input_v
        # question (rnn features)
        if 'dim_q' in self.opt:
            x_q = F.dropout(input_q, p=self.opt['dropout_q'], training=self.training)
            x_q = self.linear_q(x_q)
            if 'activation_q' in self.opt:
                x_q = getattr(F, self.opt['activation_q'])(x_q)
        else:
            x_q = input_q
        #  hadamard product
        x_mm = torch.mul(x_q, x_v)
        return x_mm


class MutanFusion(AbstractFusion):

    def __init__(self, opt, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__(opt)
        self.visual_embedding = visual_embedding
        self.question_embedding = question_embedding
        # Modules
        if self.visual_embedding:
            self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_hv'])
        else:
            print('Warning fusion.py: no visual embedding before fusion')

        if self.question_embedding:
            self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_hq'])
        else:
            print('Warning fusion.py: no question embedding before fusion')

        self.list_linear_hv = nn.ModuleList([
            nn.Linear(self.opt['dim_hv'], self.opt['dim_mm'])
            for i in range(self.opt['R'])])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(self.opt['dim_hq'], self.opt['dim_mm'])
            for i in range(self.opt['R'])])

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 2:
            raise ValueError
        batch_size = input_v.size(0)

        if self.visual_embedding:
            x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
            x_v = self.linear_v(x_v)
            if 'activation_v' in self.opt:
                x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            x_v = input_v

        if self.question_embedding:
            x_q = F.dropout(input_q, p=self.opt['dropout_q'], training=self.training)
            x_q = self.linear_q(x_q)
            if 'activation_q' in self.opt:
                x_q = getattr(F, self.opt['activation_q'])(x_q)
        else:
            x_q = input_q

        x_mm = []
        for i in range(self.opt['R']):

            x_hv = F.dropout(x_v, p=self.opt['dropout_hv'], training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)
            if 'activation_hv' in self.opt:
                x_hv = getattr(F, self.opt['activation_hv'])(x_hv)

            x_hq = F.dropout(x_q, p=self.opt['dropout_hq'], training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)
            if 'activation_hq' in self.opt:
                x_hq = getattr(F, self.opt['activation_hq'])(x_hq)

            x_mm.append(torch.mul(x_hq, x_hv))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.opt['dim_mm'])

        if 'activation_mm' in self.opt:
            x_mm = getattr(F, self.opt['activation_mm'])(x_mm)

        return x_mm


class MutanFusion2d(MutanFusion):

    def __init__(self, opt, visual_embedding=True, question_embedding=True):
        super(MutanFusion2d, self).__init__(opt,
                                            visual_embedding,
                                            question_embedding)

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 3:
            raise ValueError
        batch_size = input_v.size(0)
        weight_height = input_v.size(1)
        dim_hv = input_v.size(2)
        dim_hq = input_q.size(2)
        if not input_v.is_contiguous():
            input_v = input_v.contiguous()
        if not input_q.is_contiguous():
            input_q = input_q.contiguous()
        x_v = input_v.view(batch_size * weight_height, self.opt['dim_hv'])
        x_q = input_q.view(batch_size * weight_height, self.opt['dim_hq'])
        x_mm = super().forward(x_v, x_q)
        x_mm = x_mm.view(batch_size, weight_height, self.opt['dim_mm'])
        return x_mm
