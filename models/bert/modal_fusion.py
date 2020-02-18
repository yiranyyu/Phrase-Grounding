import torch
import math
import torch.nn as nn


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


class BilinearGrounding(nn.Module):
    def __init__(self, cfgT, cfgI):
        super(BilinearGrounding, self).__init__()
        projection = cfgI.hidden_size // 2
        self.hidden_size = projection
        self.text_hidden_size = cfgT.hidden_size
        self.imag_hidden_size = cfgI.hidden_size

        # self.Q = nn.Linear(cfgT.hidden_size, projection)
        self.K = nn.Linear(cfgI.hidden_size, cfgT.hidden_size)
        self.bilinear = nn.Bilinear(cfgT.hidden_size, cfgT.hidden_size, 1)

        # # Cannot put into memory with self.Q pre-process
        # self.bilinear = nn.Bilinear(self.text_hidden_size, self.imag_hidden_size, 1)

    def forward(self, encT: torch.Tensor, encI: torch.Tensor, mask: torch.Tensor):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        # encT = self.Q(encT)
        encI = self.K(encI)

        encT = encT.view(B, n_tok, 1, T_hidden)
        encT = encT.repeat(1, 1, n_RoI, 1)
        encT = encT.view(B, -1, T_hidden)
        encI = encI.repeat(1, n_tok, 1)

        logits = self.bilinear(encT, encI)
        logits = logits.view(B, n_tok, n_RoI)
        logits = logits + mask.squeeze(1)
        return logits


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
