import torch
import torch.nn as nn
import torch.nn.functional as F


class MLBFusion(nn.Module):
    def __init__(self,
                 T_input_hidden,
                 I_input_hidden,
                 mm_hidden_size=1200,
                 T_activate_func=torch.tanh,
                 I_activate_func=torch.tanh,
                 dropout_T=0.5,
                 dropout_I=0.5):
        """
        :param T_input_hidden: Text feature size
        :param I_input_hidden: Image feature size
        :param mm_hidden_size: multimodal hidden size
        :param T_activate_func: Text activate function, must be a callable object
        :param I_activate_func: Image activate function, must be a callable object
        :param dropout_T: rate to zero an element
        :param dropout_I: rate to zero an element
        """
        super(MLBFusion, self).__init__()

        self.linear_I = nn.Linear(I_input_hidden, mm_hidden_size)
        self.linear_T = nn.Linear(T_input_hidden, mm_hidden_size)
        self.I_activate_func = I_activate_func
        self.T_activate_func = T_activate_func
        self.dropout_I = dropout_I
        self.dropout_T = dropout_T

    def forward(self, input_T, input_I):
        """

        :param input_T: (B, ..., T_input_hidden)
        :param input_I: (B, ..., I_input_hidden), shape[:-1] must be same to that of `input_I`
        :return: (B, ..., mm_hidden)
        """
        # visual feature
        x_I = F.dropout(input_I, self.dropout_I, training=self.training)
        x_I = self.linear_I(x_I)
        if self.I_activate_func:
            x_I = self.I_activate_func(x_I)

        # text feature
        x_T = F.dropout(input_T, self.dropout_T, training=self.training)
        x_T = self.linear_T(x_T)
        if self.T_activate_func:
            x_T = self.T_activate_func(x_T)

        #  hadamard product
        x_mm = torch.mul(x_T, x_I)
        return x_mm


class MutanFusion(nn.Module):
    def __init__(self,
                 T_input_hidden,
                 I_input_hidden,
                 mm_hidden_size=510,
                 T_core_hidden=310,
                 I_core_hidden=310,
                 T_activate_func=torch.tanh,
                 I_activate_func=torch.tanh,
                 T_core_activate=torch.tanh,
                 I_core_activate=torch.tanh,
                 mm_activate=F.tanh,
                 dropout_T=0.5,
                 dropout_I=0.5,
                 dropout_core_T=0,
                 dropout_core_I=0,
                 R=5):
        """

        :param T_input_hidden: Text input hidden size
        :param I_input_hidden: Image input hidden size
        :param mm_hidden_size: output multi-modal feature size
        :param T_core_hidden: Text core hidden size
        :param I_core_hidden: Image core hidden size
        :param T_activate_func: Text activate function, should be callable
        :param I_activate_func: Image activate function, should be callable
        :param T_core_activate: Text core activate function, should be callable
        :param I_core_activate: Image core activate function, should be callable
        :param mm_activate: output multi-modal activate function
        :param dropout_T: rate to zero the element
        :param dropout_I: rate to zero the element
        :param dropout_core_T: rate to zero the element
        :param dropout_core_I: rate to zero the element
        :param R: rank of core
        """
        super(MutanFusion, self).__init__()

        self.mm_hidden_size = mm_hidden_size
        self.R = R

        self.linear_v = nn.Linear(I_input_hidden, I_core_hidden)
        self.linear_q = nn.Linear(T_input_hidden, T_core_hidden)

        self.list_linear_hv = nn.ModuleList([
            nn.Linear(I_core_hidden, mm_hidden_size)
            for _ in range(R)])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(T_core_hidden, mm_hidden_size)
            for _ in range(R)])

        self.I_activate_func = I_activate_func
        self.T_activate_func = T_activate_func
        self.I_core_activate = I_core_activate
        self.T_core_activate = T_core_activate
        self.mm_activate = mm_activate

        self.dropout_I = dropout_I
        self.dropout_T = dropout_T
        self.dropout_core_I = dropout_core_I
        self.dropout_core_T = dropout_core_T

    def forward(self, input_T, input_I):
        """
        :param input_T: (B, ..., input_T_hidden)
        :param input_I: (B, ..., input_I_hidden), shape[:-1] must be same to `input_T`
        :return: (B, ..., mm_hidden)
        """
        shape_prefix = input_I.shape[:-1]

        x_I = F.dropout(input_I, p=self.dropout_I, training=self.training)
        x_I = self.linear_v(x_I)
        if self.I_activate_func:
            x_I = self.I_activate_func(x_I)

        x_T = F.dropout(input_T, p=self.dropout_T, training=self.training)
        x_T = self.linear_q(x_T)
        if self.T_activate_func:
            x_T = self.T_activate_func(x_T)

        x_mm = []
        for i in range(self.R):

            x_hI = F.dropout(x_I, p=self.dropout_core_I, training=self.training)
            x_hI = self.list_linear_hv[i](x_hI)
            if self.I_core_activate:
                x_hI = self.I_core_activate(x_hI)

            x_hT = F.dropout(x_T, p=self.dropout_core_T, training=self.training)
            x_hT = self.list_linear_hq[i](x_hT)
            if self.T_core_activate:
                x_hT = self.T_core_activate(x_hT)

            #  hadamard product
            x_mm.append(torch.mul(x_hT, x_hI))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(*shape_prefix, self.mm_hidden_size)

        if self.mm_activate:
            x_mm = self.mm_activate(x_mm)
        return x_mm
