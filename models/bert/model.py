import torch
from pytorch_pretrained_bert import (
    BertConfig,
    BertModel,
)
from pytorch_pretrained_bert.modeling import (
    BertLayerNorm,
    BertPooler,
    BertEncoder,
    BertPreTrainedModel
)
from torch.nn import functional as F

from models.nlp import bert
from models.bert.grounding import *
from models.base.visual_context_fusion import center_dist, IoU_dist


def select(logits, target):
    """Select grounded entity RoIs by indices throughout batch.
    """
    indices, labels, types = target
    # print(f"logits={logits.shape}, indices={indices.shape}, labels={labels.shape}")
    logits = torch.cat(tuple(b[indices[i][indices[i] >= 0]] for i, b in enumerate(logits)))
    target = torch.cat(tuple(b[:(indices[i] >= 0).sum()] for i, b in enumerate(labels)))
    types = torch.cat(tuple(b[:(indices[i] >= 0).sum()] for i, b in enumerate(types)))
    entities = (indices >= 0).sum().item()
    # print(f"selected logits={logits.shape}, target={target.shape}, entities={entities}")
    return logits, target, entities, types


def bce_with_logits(logits, target, reduction='mean'):
    """
    BCE reductions:
        reduction=="mean": mean entity BCE
        reduction=="sum": batch BCE
    """
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction=reduction)
    if reduction == 'mean':
        # mean entity loss
        rois = target.shape[1]
        loss = loss * rois

    return loss


def BCE_with_logits(logits, target, mask, token_mask, cap_emb, img_emb):
    """Compute mean entity BCE of a batch.

    NOTE: mean instance BCE does not work.

    Args:
        logits (B x max_tokens x max_rois):
        target (B x max_entities x max_rois):
    """
    # weakly_loss = Align_Loss(cap_emb, img_emb, logits, mask, token_mask)[0]
    logits, target, E, _ = select(logits, target)
    bce_loss = bce_with_logits(logits, target, reduction='sum') / E

    loss = bce_loss
    return loss, E


def Align_Loss(cap_emb, img_emb, phrase_RoI_matching_score, mask, token_mask):
    margin = 0.1
    gamma = 1.0
    cap_emb = F.normalize(cap_emb, p=2, dim=-1)
    img_emb = F.normalize(img_emb, p=2, dim=-1)
    matching_score = torch.matmul(cap_emb, img_emb.transpose(-1, -2))  # (B, H) x (H, B)

    # L0 = diversity_loss(phrase_RoI_matching_score, mask, token_mask)
    # L1 = Align_Loss_prob(cap_emb, img_emb, gamma)
    L3 = Align_Loss_sum(cap_emb, img_emb, margin)

    loss = L3

    # print(f'diversity: {L0: .5f}, ranking: {L3: .5f}, loss: {loss :.5f}')
    return loss, matching_score


def assert_no_nan(tensor, name):
    assert not torch.any(torch.isnan(tensor)), f'found nan in {name} {tensor}'


def diversity_loss(phrase_RoI_matching_score, mask, token_mask):
    B, _, _ = phrase_RoI_matching_score.shape
    RoI_num = mask.sum(dim=-1)
    tok_num = token_mask.sum(dim=-1)
    epsilon = 1E-8

    loss = torch.tensor(0.).cuda()
    for i in range(B):
        score = torch.softmax(phrase_RoI_matching_score[i, :tok_num[i], :RoI_num[i]], dim=-1).clamp(min=epsilon)
        assert_no_nan(score, 'score.softmax')
        score = score * score.log()
        assert_no_nan(score, 'score * score.log')
        loss += score.sum()
    loss = -loss
    loss /= max(1, tok_num.sum())
    return loss


def Align_Loss_prob(cap_emb, img_emb, gamma=10.0):
    matching_score = torch.matmul(cap_emb, img_emb.transpose(-1, -2))  # (B, H) x (H, B)
    matching_score *= gamma
    P_CI = torch.diag(torch.softmax(matching_score, dim=1))
    P_IC = torch.diag(torch.softmax(matching_score, dim=0))
    L1 = -torch.mean(torch.log(P_CI))
    L2 = -torch.mean(torch.log(P_IC))
    return L1 + L2


def Align_Loss_sum(cap_emb, img_emb, margin=0.1):
    matching_score = torch.matmul(cap_emb, img_emb.transpose(-1, -2))  # (B, H) x (H, B)
    paired_score = torch.diag(matching_score)

    B, _ = matching_score.shape
    matching_score[torch.arange(B), torch.arange(B)] = -10

    loss_I_C = torch.sum(torch.relu(matching_score - paired_score.unsqueeze(0).repeat(B, 1) + margin), dim=0)
    loss_I_C = torch.mean(loss_I_C)

    loss_C_I = torch.sum(torch.relu(matching_score - paired_score.unsqueeze(1).repeat(1, B) + margin), dim=1)
    loss_C_I = torch.mean(loss_C_I)

    loss = loss_C_I + loss_I_C
    return loss


# Pooling_based_Loss
def Pooling_based_Loss(encT, encI, token_mask, mask):
    B, n_tok = token_mask.shape
    _, n_RoI = mask.shape
    n = 5
    margin = 0.1

    # POOLING BASED
    fragment_score = torch.matmul(F.normalize(encT.view(B * n_tok, -1), p=2, dim=-1),
                                  F.normalize(encI.view(B * n_RoI, -1), p=2, dim=-1).transpose(-1, -2))
    fragment_tok_mask = token_mask.view(-1)
    fragment_RoI_mask = mask.view(-1)
    fragment_score *= fragment_tok_mask.unsqueeze(1) * fragment_RoI_mask.unsqueeze(0)
    fragment_score = torch.relu(fragment_score)

    S = fragment_score.new_zeros((B, B))
    for i in range(B):
        for j in range(B):
            row_st = i * n_tok
            col_st = j + n_RoI
            S[i, j] = torch.sum(fragment_score[row_st: row_st + n_tok, col_st: col_st + n_RoI])

    num_RoI = mask.sum(dim=1).float()
    num_pair = torch.matmul(token_mask.sum(dim=1).float().unsqueeze(1) + n, num_RoI.unsqueeze(0) + n)
    S /= num_pair
    paired_score = torch.diag(S)

    loss_I_C = torch.sum(torch.relu(S - paired_score.unsqueeze(0).repeat(B, 1) + margin), dim=0)
    loss_I_C = torch.mean(loss_I_C)

    loss_C_I = torch.sum(torch.relu(S - paired_score.unsqueeze(1).repeat(1, B) + margin), dim=1)
    loss_C_I = torch.mean(loss_C_I)

    loss = loss_C_I + loss_I_C
    return loss

    # END POOLING BASED


def Align_Loss_real(cap_emb, img_emb, margin=0.1):
    matching_score = torch.matmul(cap_emb, img_emb.transpose(-1, -2))  # (B, H) x (H, B)
    paired_score = torch.diag(matching_score)

    B, _ = matching_score.shape
    matching_score[torch.arange(B), torch.arange(B)] = -10

    loss_I_C = torch.max(torch.relu(matching_score - paired_score.unsqueeze(0).repeat(B, 1) + margin), dim=0)[0]
    loss_I_C = torch.mean(loss_I_C)

    loss_C_I = torch.max(torch.relu(matching_score - paired_score.unsqueeze(1).repeat(1, B) + margin), dim=1)[0]
    loss_C_I = torch.mean(loss_C_I)

    loss = loss_C_I + loss_I_C
    return loss


# noinspection PyArgumentList
def recall(logits, target, topk=(1, 5, 10), typeN=8):
    """Compute top K recalls of a batch.

    Args:
        logits (B x max_entities, B x max_entities x max_rois):
        target (B x max_entities, B x max_entities x max_rois):
        topk: top k recalls to compute
        typeN: number of types

    Returns:
        N: number of entities in the batch
        TPs: topk true positives in the batch
        bound: max number of groundable entities
    """

    logits, target, N, types = select(logits, target)
    topk = [topk] if isinstance(topk, int) else sorted(topk)
    TPs = [0] * len(topk)
    bound = target.max(-1, False)[0].sum().item()  # at least one detected
    typeTPs = torch.zeros(typeN, device=types.device)
    typeN = torch.zeros_like(typeTPs)

    # print("target entity type count: ", types.shape, types.sum(dim=0), target.shape)
    if max(topk) == 1:
        top1 = torch.argmax(logits, dim=1)
        one_hots = torch.zeros_like(target)
        one_hots.scatter_(1, top1.view(-1, 1), 1)
        TPs = (one_hots * target).sum().item()
        hits = (one_hots * target).sum(dim=1) >= 1
        typeTPs += types[hits].sum(dim=0)
        typeN += types.sum(dim=0)
    else:
        logits = torch.sort(logits, 1, descending=True)[1]
        for i, k in enumerate(topk):
            one_hots = torch.zeros_like(target)
            one_hots.scatter_(1, logits[:, :k], 1)
            TPs[i] = ((one_hots * target).sum(dim=1) >= 1).float().sum().item()  # hit if at least one matched
            if i == 0:
                hits = (one_hots * target).sum(dim=1) >= 1
                typeTPs += types[hits].sum(dim=0)
                typeN += types.sum(dim=0)

    # print(TPs, N)
    # print(typeTPs)
    # print(typeN)
    return N, torch.Tensor(TPs + [bound]), (typeTPs.cpu(), typeN.cpu())


class IBertConfig(BertConfig):
    def __init__(
            self,
            vocab_size_or_config_json_file=-1,
            hidden_size=2048,  # 768,
            num_hidden_layers=3,  # 3, 12,
            num_attention_heads=2,  # 2, 12,
            intermediate_size=3072,  # 3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.5,  # 0.1,
            attention_probs_dropout_prob=0.5,  # 0.1,
            max_position_embeddings=100,  # 512,
            type_vocab_size=2,
            initializer_range=0.02,
            spatial=None,
    ):
        super(IBertConfig, self).__init__(
            vocab_size_or_config_json_file,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            initializer_range,
        )
        self.spatial = spatial


class SpatialEmbedding(nn.Module):
    def __init__(self, din, dout, hidden=None, relative=False):
        super(SpatialEmbedding, self).__init__()
        hidden = hidden or dout * 2
        self.mlp = nn.Sequential(nn.Linear(din, hidden), nn.Linear(hidden, dout))
        self.relative = relative
        # XXX Never override forward() in case of data parallelism

    def forward_relative(self, spatials):
        raise NotImplementedError(f"Relative spatial encoding not implemented yet")

    def forward_absolute(self, spatials):
        bs, rois, dim = spatials.shape
        return self.mlp(spatials)

    def forward(self, spatials):
        return self.forward_relative(spatials) if self.relative else self.forward_absolute(spatials)


class IBertEmbeddings(nn.Module):
    def __init__(self, config, din=6):
        super(IBertEmbeddings, self).__init__()
        self.spatial_embedding = SpatialEmbedding(
            din, config.hidden_size, hidden=2 * config.hidden_size, relative=config.spatial == 'rel'
        )
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, spatials):
        embeddings = self.spatial_embedding(spatials)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class IBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(IBertModel, self).__init__(config)
        self.embeddings = None if config.spatial is None else IBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
            self, features, spatials, attention_mask=None, output_all_encoded_layers=True
    ):
        if attention_mask is None:
            attention_mask = torch.ones(features.shape[0:2])

        # rois = attention_mask.sum(dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = features if self.embeddings is None else features + self.embeddings(spatials)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        rois_output = encoded_layers[-1]

        # TODO: extract whole scene features
        pooled_output = self.pooler(rois_output.mean(dim=1, keepdim=True))
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForGrounding(nn.Module):
    # noinspection PyUnresolvedReferences
    def __init__(self, cfgI):
        super(BertForGrounding, self).__init__()
        bert.setup(base=True, uncased=True)
        self.tBert = BertModel.from_pretrained(bert.pretrained())
        self.iBert = IBertModel(cfgI)
        self.grounding = CosineGrounding(
            self.tBert.config,
            self.iBert.config,
            use_neighbor=False,
            use_global=False,
            k=1,
            dist_func=center_dist)

        # self.k = 3
        # self.hidden = 256
        # self.Qs = nn.Sequential(nn.Linear(768 * 2, self.hidden),
        #                         nn.ReLU(),
        #                         nn.Linear(self.hidden, self.hidden))
        #
        # self.deep_set = nn.Sequential(nn.Linear(2048, self.hidden),
        #                               nn.ReLU(),
        #                               nn.Linear(self.hidden, self.hidden))

    # noinspection PyTypeChecker
    def forward(self, batch):
        features, global_ctx, spatials, mask, token_ids, token_segs, token_mask, phrase_indices = batch
        encT, _ = self.tBert(token_ids, token_segs, token_mask, output_all_encoded_layers=False)

        if mask is None:
            mask = torch.ones(features.shape[0:2])
        extended_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        encI, _ = self.iBert(features, spatials, mask, output_all_encoded_layers=False)
        matching_score = self.grounding(encT, encI, extended_mask, None, None)

        # B, n_tok, n_RoI = matching_score.shape
        # topk_idx = matching_score.topk(k=self.k, dim=-1)[1]
        # lengths = token_mask.sum(dim=1).long()
        #
        # # (B, n_tok, n_RoI)
        # topk_mask = torch.zeros_like(matching_score).scatter(dim=-1, index=topk_idx, value=1)
        #
        # # (B, n_tok, 1)
        # matched_idx = topk_mask.new_zeros((B, n_tok, 1))
        # for i in range(B):
        #     matched_idx[i] = torch.multinomial(topk_mask[i], 1)
        # matched_mask = torch.zeros_like(topk_mask).scatter(dim=-1, index=matched_idx.long(), value=1)
        #
        # # (B, n_tok, H)
        # matched_RoI_feat = (matched_mask.unsqueeze(-1) * encI.unsqueeze(1)).sum(dim=2)
        # matched_RoI_feat = (phrase_indices.unsqueeze(-1) * matched_RoI_feat).sum(dim=1)
        # matched_RoI_feat = matched_RoI_feat / torch.clamp(phrase_indices.sum(dim=1, keepdim=True), min=1)
        # img_emb = self.deep_set(matched_RoI_feat)
        #
        # # (B, hidden)
        # sent_st = encT[:, 0]
        # sent_ed = encT[torch.arange(B), lengths - 1]
        # cap_emb = self.Qs(torch.cat([sent_st, sent_ed], dim=-1))

        return matching_score, [], [], token_mask, mask
        # return matching_score, img_emb, cap_emb, token_mask, mask

    # noinspection PyUnresolvedReferences
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        states = super(BertForGrounding, self).state_dict(destination, prefix, keep_vars)
        states["tBert.config"] = self.tBert.config,
        states["iBert.config"] = self.iBert.config,
        return states

    def load_state_dict(self, states, strict=False):
        super().load_state_dict(states, strict=strict)
        self.tBert.config = states["tBert.config"]
        self.iBert.config = states["iBert.config"]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        states = torch.load(path)
        self.load_state_dict(states)
