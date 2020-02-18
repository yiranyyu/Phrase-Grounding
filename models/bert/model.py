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
from models.bert.modal_fusion import *


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


def BCE_with_logits(logits, target):
    """Compute mean entity BCE of a batch.

    NOTE: mean instance BCE does not work.

    Args:
        logits (B x max_tokens x max_rois):
        target (B x max_entities x max_rois):
    """
    logits, target, E, _ = select(logits, target)
    return bce_with_logits(logits, target, reduction='sum') / E, E


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
    """
    TODO:
        Sence embedding with spatial cfg: (0, 0, 1, 1, 1, 1)
    """

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
    def __init__(self, cfgI):
        super(BertForGrounding, self).__init__()
        bert.setup(base=True, uncased=True)
        self.tBert = BertModel.from_pretrained(bert.pretrained())
        self.iBert = IBertModel(cfgI)
        self.grounding = ModalFusionConcatenateGrounding(self.tBert.config, self.iBert.config)

    def forward(self, batch):
        features, spatials, mask, token_ids, token_segs, token_mask = batch
        encT, _ = self.tBert(token_ids, token_segs, token_mask, output_all_encoded_layers=False)

        if mask is None:
            mask = torch.ones(features.shape[0:2])
        extended_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        encI, _ = self.iBert(features, spatials, mask, output_all_encoded_layers=False)
        output = self.grounding(encT, encI, extended_mask)
        return output

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
