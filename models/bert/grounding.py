import pickle
from typing import Dict, Any, Tuple, Callable, Union

import torch
import math
import torch.nn as nn
from models.base.modal_fusion import MLBFusion, MutanFusion
from models.NMTree.tree import SingleScore, UpTreeLSTM, DownTreeLSTM, PairScore, build_bitree
from models.base.visual_context_fusion import VisualContextFusion, center_dist, IoU_dist
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
    def __init__(self, cfgT, cfgI, heads=1, use_neighbor=False, use_global=False, k=5, dist_func=center_dist):
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

        self.use_neighbor = use_neighbor
        self.use_global = use_global
        # self.visual_context_fusion = VisualContextFusion(cfgI, k, self.use_neighbor, self.use_global, dist_func)
        logging.info(f'Grounding using {self.__class__.__name__}'
                     f'{"N" if use_neighbor else ""} {"G" if use_global else ""}')


class CosineGrounding(AbstractGrounding):
    def __init__(self, cfgT, cfgI, heads=1, use_neighbor=False, use_global=False, k=5, dist_func=center_dist):
        super(CosineGrounding, self).__init__(cfgT, cfgI, heads, use_neighbor, use_global, k, dist_func)

        self.Q = nn.Linear(cfgT.hidden_size, self.all_head_size)
        self.K = nn.Linear(cfgI.hidden_size, self.all_head_size)
        # self.K_ng = nn.Linear(cfgI.hidden_size, self.all_head_size)

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # B x #tokens x #heads x head_size
        return x.permute(0, 2, 1, 3)  # B x #heads x # tokens x head_size

    def forward(self, encT, encI, mask, spatials=None, global_ctx=None):
        assert global_ctx is not None or not self.use_global
        assert spatials is not None or not self.use_neighbor

        # neighbor = self.visual_context_fusion(encI, 1 + (mask[:, 0, 0, :] / 10000), spatials, global_ctx)
        # neighbor = self.K_ng(neighbor)
        # neighbor = self.transpose(neighbor)

        Q = self.Q(encT)
        K = self.K(encI)
        Q = self.transpose(Q)
        K = self.transpose(K)

        logits = torch.matmul(Q, K.transpose(-1, -2))
        # neighbor_logits = torch.matmul(Q, neighbor.transpose(-1, -2))
        # logits += neighbor_logits
        logits = logits / math.sqrt(self.attention_head_size)
        logits = logits + mask
        return logits.squeeze()


class NMTGrounding(AbstractGrounding):
    def __init__(self, cfgT, cfgI, heads=1):
        super(NMTGrounding, self).__init__(cfgT, cfgI, heads)

        self.Q = nn.Linear(cfgT.hidden_size, 300)
        self.K = nn.Linear(cfgI.hidden_size, self.all_head_size)

        cache1 = 'data/flickr30k_entities/test_entities_tree1.pt'
        info = torch.load(cache1)
        self.word_vocab_size = len(info['word_to_ix'])
        self.tag_vocab_size = len(info['tag_to_ix'])
        self.dep_vocab_size = len(info['dep_to_ix'])
        self.rnn_size = 1024
        self.vis_dim = 2048 + 512 + 512
        self.embed_size = 300 + 50 + 50

        self.word_embedding = nn.Embedding(num_embeddings=self.word_vocab_size,
                                           embedding_dim=300)
        self.tag_embedding = nn.Embedding(num_embeddings=self.tag_vocab_size,
                                          embedding_dim=50)
        self.dep_embedding = nn.Embedding(num_embeddings=self.dep_vocab_size,
                                          embedding_dim=50)

        self.word_vocab = info['word_to_ix']
        self.tag_vocab = info['tag_to_ix']
        self.dep_vocab = info['dep_to_ix']

        self.dropout = nn.Dropout(0.5)

        self.single_score = SingleScore(self.vis_dim, self.embed_size)
        self.pair_score = PairScore(self.vis_dim, self.embed_size)

        self.up_tree_lstm = UpTreeLSTM(self.embed_size, self.rnn_size)
        self.down_tree_lstm = DownTreeLSTM(self.embed_size, self.rnn_size)

        self.sbj_attn_logit = nn.Linear(self.rnn_size * 2, 1)
        self.rlt_attn_logit = nn.Linear(self.rnn_size * 2, 1)
        self.obj_attn_logit = nn.Linear(self.rnn_size * 2, 1)

    def traversal(self, node, v, embedding):
        node.logit = self.node_to_logit(node)
        for idx in range(node.num_children):
            self.traversal(node.children[idx], v, embedding)

        # Case 1: Leaf, update node.sub_word and node.score
        if node.num_children == 0:
            node.sub_word = [node.idx]
            node.sub_logit = [node.logit]
            sbj_embedding = self.attn_embedding(node.sub_word, node.sub_logit, embedding, 'sbj')
            sbj_score = self.single_score(v, sbj_embedding)
            node.score = sbj_score
        # Case 2: Not leaf
        else:
            sub_word = [node.idx]
            sub_logit = [node.logit]
            for c in node.children:
                sub_word = sub_word + c.sub_word
                sub_logit = sub_logit + c.sub_logit

            # Case 2.1: Not leaf, Language Node
            sbj_score = self.zero_score
            for c in node.children:
                sbj_score = sbj_score + c.score

            # Case 2.2: Not leaf, Visual Node
            obj_embedding = self.attn_embedding(sub_word, sub_logit, embedding, 'obj')
            obj_score = self.single_score(v, obj_embedding)
            for c in node.children:
                obj_score = obj_score + c.score
            node.obj_score = obj_score
            rlt_embedding = self.attn_embedding(sub_word, sub_logit, embedding, 'rlt')
            rlt_score = self.pair_score(v, obj_score, rlt_embedding)

            # Gumbel softmax, [lang, vis]
            node.sub_word = sub_word
            node.sub_logit = sub_logit
            node.score = torch.mm(node.type, torch.stack([sbj_score, rlt_score], dim=0)).squeeze(0)

        return

    def list_to_embedding(self, sent):
        """
        translate a sentence into embedding
        Input: list of sentence, contains tokens, tags, deps
        Output: embedding (num_words * embedding_size)
        """
        word_ids = []
        for word in sent['tokens']:
            word_id = self.word_vocab[word] if word in self.word_vocab else self.word_vocab['UNK']
            word_ids.append(word_id)

        tag_ids = []
        for tag in sent['tags']:
            tag_id = self.tag_vocab[tag] if tag in self.tag_vocab else self.tag_vocab['UNK']
            tag_ids.append(tag_id)

        dep_ids = []
        for dep in sent['deps']:
            dep_id = self.dep_vocab[dep] if dep in self.dep_vocab else self.dep_vocab['UNK']
            dep_ids.append(dep_id)

        word_ids = torch.tensor(word_ids).cuda()
        w_embed = self.word_embedding(word_ids)
        tag_ids = torch.tensor(tag_ids).cuda()
        t_embed = self.tag_embedding(tag_ids)
        dep_ids = torch.tensor(dep_ids).cuda()
        d_embed = self.dep_embedding(dep_ids)

        embedding = torch.cat([w_embed, t_embed, d_embed], dim=-1)
        return self.dropout(embedding)

    def attn_embedding(self, word_list, logit_list, embedding, logit_type):
        embed = embedding[word_list]
        logits = torch.stack([s[logit_type] for s in logit_list], dim=0)

        attn = torch.softmax(logits, dim=0)
        attn_embed = torch.mm(attn.unsqueeze(0), embed)

        return attn_embed

    def node_to_logit(self, node):
        hidden = torch.cat([node.up_state[1], node.down_state[1]], dim=-1)
        sbj_logit = self.sbj_attn_logit(hidden).squeeze()
        rlt_logit = self.rlt_attn_logit(hidden).squeeze()
        obj_logit = self.obj_attn_logit(hidden).squeeze()

        logit = {'sbj': sbj_logit, 'rlt': rlt_logit, 'obj': obj_logit}

        return logit

    def transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # B x #tokens x #heads x head_size
        return x.permute(0, 2, 1, 3)  # B x #heads x # tokens x head_size

    @staticmethod
    def remapping_word_feature(encT, word2piece_mapping):
        # N word-pieces, K words
        # Bert -> (B, N, hidden)
        # -> (B, K, hidden)
        # TODO: map bert-tokenized token seq to spacy DPT token seq, merge features of word-pieces in same word
        return encT

    def forward(self, encT, encI, mask, tree_data_batch, word2piece_mapping):
        B, n_RoI, I_hidden = encI.shape
        _, n_tok, T_hidden = encT.shape

        Q = self.Q(encT)
        K = self.K(encI)
        Q = self.remapping_word_feature(Q, word2piece_mapping)

        tree_dict_batch = pickle.loads(tree_data_batch[0])
        logits = Q.new_zeros((B, n_tok, n_RoI))

        # TODO: process every (caption, img) pair chronologically
        for item_idx, tree_dict in enumerate(tree_dict_batch):
            tree = build_bitree(tree_dict)
            self.up_tree_lstm(tree, Q[item_idx])
            self.down_tree_lstm(tree, Q[item_idx])
            self.traversal(tree, K, Q[item_idx])

            # TODO: collect matching score of every word from tree

            # TODO: map logits of words to logits of word-pieces
            logits[item_idx] = ...

        logits = logits / math.sqrt(self.attention_head_size)
        logits = logits + mask
        return logits.squeeze()


class LinearSumGrounding(AbstractGrounding):
    def __init__(self, cfgT, cfgI):
        super(LinearSumGrounding, self).__init__(cfgT, cfgI)

        self.Q_mlp = nn.Sequential(
            nn.Linear(cfgT.hidden_size, self.projection),
            nn.GELU(),
            # nn.Linear(self.projection, self.projection),
            # nn.Dropout()
        )
        self.K_mlp = nn.Sequential(
            nn.Linear(cfgI.hidden_size, self.projection),
            nn.GELU(),
            # nn.Linear(self.projection, self.projection),
            # nn.Dropout()
        )
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
            # nn.ReLU(),
            # nn.GELU(),
            # nn.Linear(self.projection, self.projection),
            # nn.Dropout()
        )
        self.K_mlp = nn.Sequential(
            nn.Linear(cfgI.hidden_size, self.projection),
            # nn.ReLU(),
            # nn.GELU(),
            # nn.Linear(self.projection, self.projection),
            # nn.Dropout()
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.projection * 2, self.projection),
            nn.ReLU(),
            # nn.GELU(),
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


class CosineCrossModalSupervisionGrounding(nn.Module):
    def __init__(self, cfgT, cfgI):
        super(CosineCrossModalSupervisionGrounding, self).__init__()

        self.projection = cfgI.hidden_size // 2
        self.Q = nn.Linear(cfgT.hidden_size, self.projection)
        self.K = nn.Linear(cfgI.hidden_size, self.projection)

    def forward(self, encT, encI, mask):
        """
        :param encT: (B, n_tok, T_hidden), encoded text feature
        :param encI: (B, n_RoI, I_hidden) encoded image feature
        :param mask:
        :return: logits
        """
        Q = self.Q(encT)
        K = self.K(encI)

        attention = torch.matmul(Q, K.transpose(-1, -2))
        attention = attention / math.sqrt(self.projection)

        # (B, n_RoI) -> (B, 1, n_RoI)
        I_supervision = (attention.softmax(dim=1) * attention).sum(dim=1)
        I_supervision = I_supervision.unsqueeze(1)

        logits = attention + I_supervision + mask.squeeze(1)
        return logits


class FusionFusionGrounding(AbstractGrounding):
    # To suppress IDE warning
    att_fusion: Union[Tuple[Callable[[Tuple[Any, ...], Dict[str, Any]], Any]], Callable]
    cls_fusion: Union[Tuple[Callable[[Tuple[Any, ...], Dict[str, Any]], Any]], Callable]

    def __init__(self,
                 cfgT,
                 cfgI,
                 attention_fusion=MLBGrounding,
                 classification_fusion=MLBGrounding):
        super(FusionFusionGrounding, self).__init__(cfgT, cfgI)

        self.attention = attention_fusion(cfgT, cfgI)

        cfgT.hidden_size = self.text_hidden_size + self.imag_hidden_size
        cfgI.hidden_size = self.text_hidden_size + self.imag_hidden_size
        self.classification = classification_fusion(cfgT, cfgI)

    def forward(self, encT, encI, mask, spatials):
        # (B, n_tok, n_RoI)
        attention = self.attention(encT, encI, mask, None)
        attention_on_T: torch.Tensor = attention.softmax(dim=1)
        attention_on_I: torch.Tensor = attention.softmax(dim=2)

        attented_T = torch.bmm(attention_on_T.permute(0, 2, 1), encT)
        attented_I = torch.bmm(attention_on_I, encI)
        fused_T = torch.cat([encT, attented_I], dim=-1)
        fused_I = torch.cat([encI, attented_T], dim=-1)

        logits = self.classification(fused_T, fused_I, mask, spatials)
        logits = logits + attention + mask.squeeze(1)
        return logits
