import torch
import torch.nn as nn
from util import logging

inf = 3000


def tensor_intersect(box_a, box_b):
    """
    :param box_a: (B, N, 4), as normalized (x1, y1, x2, y2)
    :param box_b: (B, M, 4), as normalized (x1, y1, x2, y2)
    :return: (B, N, M) Intersection area.
    """
    B = box_a.size(0)
    N = box_a.size(1)
    M = box_b.size(1)
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(B, N, M, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(B, N, M, 2))
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(B, N, M, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(B, N, M, 2))

    # inter[:, :, : 0] is the width of intersection and inter[:, :, :, 1] is height
    inter = torch.clamp((max_xy - min_xy), min=0)
    area = inter[:, :, :, 0] * inter[:, :, :, 1]
    return area


def tensor_area(bbox):
    """
    :param bbox: (B, n_RoI, 4)
    :return: (B, n_RoI) area
    """
    x1, x2 = bbox[:, :, 0], bbox[:, :, 2]
    y1, y2 = bbox[:, :, 1], bbox[:, :, 3]
    area = (x2 - x1) * (y2 - y1)
    return area


def tensor_IoU(A, B):
    """
    Compute the IoU of two sets of boxes.
    A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    Args:
        A: (B, N, 4), normalized bbox (xmin, ymin, xmax, ymax)
        B: (B, M, 4), normalized bbox (xmin, ymin, xmax, ymax)
    :return (B, N, M)
    """

    # (B, N, M)
    inter = tensor_intersect(A, B)

    area_A = tensor_area(A).unsqueeze(2).expand_as(inter)
    area_B = tensor_area(B).unsqueeze(1).expand_as(inter)

    union = area_A + area_B - inter
    IoU = inter / union
    return IoU


def IoU_dist(spatial, mask):
    """
    :param spatial: (B, n_RoI, 4)
    :param mask: (B, n_RoI)
    :return: (B, n_RoI, n_RoI) between bboxes
    """
    B, n_RoI, _ = spatial.shape
    box = spatial[:, :, :4]
    IoU = tensor_IoU(box, box)  # (B, n_RoI, n_RoI)

    dist = 1.0 - IoU
    dist[:, torch.arange(n_RoI), torch.arange(n_RoI)] = inf
    dist = dist.where(mask.byte().unsqueeze(1).expand(B, n_RoI, n_RoI), torch.zeros_like(dist) + inf)
    # torch.set_printoptions(profile="full")
    # print(f'dist={dist[:, :20, :20]}')
    # torch.set_printoptions(profile="default")
    return dist


def center_dist(spatial, mask):
    """
    :param spatial: (B, n_RoI, 6), normalized (x1, y1, x2, y2, w, h)
    :param mask: (B, n_RoI)
    :return: (B, n_RoI, n_RoI) distances among boxes
    """
    B, n_RoI, _ = spatial.shape
    center = spatial.new_empty(size=(B, n_RoI, 2))

    center[:, :, 0] = (spatial[:, :, 0] + spatial[:, :, 2]) / 2
    center[:, :, 1] = (spatial[:, :, 1] + spatial[:, :, 3]) / 2

    center = center.where(mask.byte().unsqueeze(-1).expand(B, n_RoI, 2), center + inf)
    a = center.view(B, n_RoI, 1, 2).repeat(1, 1, n_RoI, 1)
    b = center.view(B, 1, n_RoI, 2).repeat(1, n_RoI, 1, 1)
    distance = (a - b).norm(dim=3)
    distance[:, torch.arange(n_RoI), torch.arange(n_RoI)] = inf * inf
    return distance


def gen_nearest_mask(spatial, mask, k, dist_func) -> torch.Tensor:
    """

    :param k:
    :param dist_func:
    :param spatial: (B, n_RoI, 6), normalized (x1, y1, x2, y2, w, h)
    :param mask: (B, n_RoI) 1 for RoI, 0 for padding
    :return: (B, n_RoI, n_RoI) mask, 1 for topk nearest RoI. Assured that self-mask is 0, padding-mask is 0
             Since padding and self mask are bound to be 0, number of 1 in a row might be less than k
    """
    B, n_RoI, _ = spatial.shape
    distance = dist_func(spatial, mask)
    val, idx = distance.topk(k, dim=-1, largest=False)
    topk_mask = torch.zeros_like(distance).scatter(dim=-1, index=idx, src=torch.ones_like(val))
    topk_mask[:, torch.arange(n_RoI), torch.arange(n_RoI)] = 0
    topk_mask = topk_mask.where(mask.byte().unsqueeze(1).expand(B, n_RoI, n_RoI), torch.zeros_like(topk_mask))
    return topk_mask


class VisualContextFusion(nn.Module):
    def __init__(self, cfgI, k, use_neighbor, use_global, dist_func=center_dist):
        super(VisualContextFusion, self).__init__()
        self.k = k
        logging.info(f'Neighboring k={k}')

        self.dist_func = dist_func
        self.use_global = int(use_global)
        self.use_neighbor = int(use_neighbor)
        self.fc = nn.Linear(cfgI.hidden_size * (1 + self.use_neighbor + self.use_global), cfgI.hidden_size)

    def forward(self, encI, RoI_mask, spatials=None, global_ctx=None):
        """
        :param global_ctx: (B, I_hidden)
        :param encI: (B, n_RoI, I_hidden) Encoded RoI features
        :param RoI_mask: (B, n_RoI) 1 if kth RoI exists
        :param spatials: (B, n_RoI, 6) Spatial feature as normalized (x1, y1, x2, y2, w, h)
        :return: (B, n_RoI, I_hidden) fused RoI features
        """
        assert global_ctx is not None or not self.use_global
        assert spatials is not None or not self.use_neighbor

        B, n_RoI, I_hidden = encI.shape
        if spatials is None:
            if global_ctx is None:
                return encI
            else:
                return self.fc(torch.cat([encI, global_ctx.unsqueeze(1).repeat(1, n_RoI, 1)], dim=-1))

        similarity = torch.matmul(encI, encI.transpose(-1, -2))
        similarity[:, torch.arange(n_RoI), torch.arange(n_RoI)] = 0

        # 1 for topk nearest RoIs, 0 otherwise
        topk_mask = gen_nearest_mask(spatials, RoI_mask, self.k, self.dist_func)
        topk_weight = similarity.where(topk_mask.byte(), torch.zeros_like(similarity))
        topk_weight = topk_weight.softmax(dim=-1).unsqueeze(-1)

        neighbor_context = (topk_weight * encI.unsqueeze(1)).sum(dim=2)

        if global_ctx is None:
            fused = torch.cat([encI, neighbor_context], dim=-1)
        else:
            fused = torch.cat([encI, neighbor_context, global_ctx], dim=-1)
        fused = self.fc(fused)
        return fused
