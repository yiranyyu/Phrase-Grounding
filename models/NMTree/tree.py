import torch
import torch.nn as nn
import torch.nn.functional as F

from models.NMTree import gumbel


class SingleScore(nn.Module):
    """Compute a unsoftmax similarity between single region and language"""

    def __init__(self, vis_dim, word_size):
        super(SingleScore, self).__init__()

        self.W = nn.Linear(vis_dim, word_size)
        self.fc = nn.Linear(word_size, 1)

    def forward(self, v, w):
        """
        v : vis:    Tensor float (num_bbox, vis_dim)
        w : embed:  Tensor float (word_size, )
        l : logit:  Tensor float (num_bbox, )
        """
        v = self.W(v)
        w = w.expand_as(v)
        f = nn.functional.normalize(v * w, p=2, dim=1)
        l = self.fc(f).squeeze(1)

        return l


class PairScore(nn.Module):
    """Compute a unsoftmax similarity between pairwise regions and language"""

    def __init__(self, vis_dim, word_size):
        super(PairScore, self).__init__()

        self.W = nn.Linear(vis_dim * 2, word_size)
        self.fc = nn.Linear(word_size, 1)

    def forward(self, v, l, w):
        """
        v   : vis:    Tensor float (num_bbox, vis_dim)
        l   : logit:  Tensor float (num_bbox, )
        w   : embed:  Tensor float (word_size, )
        l_2 : logit:  Tensor float (num_bbox, )
        """
        s = F.softmax(l, 0)
        v_2 = torch.mm(s.unsqueeze(0), v)

        v_3 = torch.cat((v, v_2.repeat(v.size(0), 1)), dim=1)
        v_3 = self.W(v_3)
        w = w.expand_as(v_3)
        f = nn.functional.normalize(v_3 * w, p=2, dim=1)
        l_2 = self.fc(f).squeeze(1)

        return l_2


class UpTreeLSTM(nn.Module):
    """
    Adapted from:
    https://github.com/dasguptar/treelstm.pytorch/blob/master/treelstm/model.py
    """

    def __init__(self, in_dim, mem_dim):
        super(UpTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        # gates for visual node
        self.ioux_vis = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_vis = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx_vis = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_vis = nn.Linear(self.mem_dim, self.mem_dim)

        # gates for language node
        self.ioux_lang = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_lang = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx_lang = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_lang = nn.Linear(self.mem_dim, self.mem_dim)

        # Gumbel softmax
        self.type_query = nn.Linear(self.in_dim, 2)
        self.dropout = nn.Dropout()

    def node_forward_vis(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux_vis(inputs) + self.iouh_vis(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh_vis(child_h) +
            self.fx_vis(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def node_forward_lang(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux_lang(inputs) + self.iouh_lang(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh_lang(child_h) +
            self.fx_lang(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].data.new(1, self.mem_dim).zero_()
            child_h = inputs[0].data.new(1, self.mem_dim).zero_()
        else:
            child_c, child_h = zip(*map(lambda x: x.up_state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        c_vis, h_vis = self.node_forward_vis(inputs[tree.idx], child_c, child_h)
        c_lang, h_lang = self.node_forward_lang(inputs[tree.idx], child_c, child_h)

        # Gumbel softmax decision
        type_value = inputs[tree.idx].unsqueeze(0)
        type_weights = self.type_query(type_value)

        if self.training:
            type_mask = gumbel.st_gumbel_softmax(logits=type_weights)
        else:
            type_mask = gumbel.greedy_select(logits=type_weights)
            type_mask = type_mask.float()

        h = torch.mm(type_mask, torch.cat([h_lang, h_vis], dim=0))
        c = torch.mm(type_mask, torch.cat([c_lang, c_vis], dim=0))

        tree.type = type_mask
        tree.type_ = 'l' if type_mask[0, 0].item() else 'v'
        if tree.num_children == 0:
            tree.type_ = 'l'

        tree.up_state = (c, self.dropout(h))

        return tree.up_state


class DownTreeLSTM(nn.Module):
    """
    Adapted from:
    https://github.com/dasguptar/treelstm.pytorch/blob/master/treelstm/model.py
    """

    def __init__(self, in_dim, mem_dim):
        super(DownTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        # gates for visual node
        self.ioux_vis = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_vis = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx_vis = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_vis = nn.Linear(self.mem_dim, self.mem_dim)

        # gates for language node
        self.ioux_lang = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_lang = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx_lang = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_lang = nn.Linear(self.mem_dim, self.mem_dim)

        # Gumbel softmax
        self.dropout = nn.Dropout()

    def node_forward_vis(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux_vis(inputs) + self.iouh_vis(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh_vis(child_h) +
            self.fx_vis(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def node_forward_lang(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux_lang(inputs) + self.iouh_lang(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh_lang(child_h) +
            self.fx_lang(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        if tree.parent == None:
            child_c = inputs[0].data.new(1, self.mem_dim).zero_()
            child_h = inputs[0].data.new(1, self.mem_dim).zero_()
        else:
            child_c, child_h = tree.parent.down_state

        c_vis, h_vis = self.node_forward_vis(inputs[tree.idx], child_c, child_h)
        c_lang, h_lang = self.node_forward_lang(inputs[tree.idx], child_c, child_h)

        # Gumbel softmax decision
        type_mask = tree.type
        h = torch.mm(type_mask, torch.cat([h_lang, h_vis], dim=0))
        c = torch.mm(type_mask, torch.cat([c_lang, c_vis], dim=0))

        tree.down_state = (c, self.dropout(h))

        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        return tree.down_state


class BiTree(object):
    def __init__(self, wd):
        # structure
        idx, word, pos, tag, dep = wd
        self.pos = pos
        self.tag = tag
        self.dep = dep
        self.word = word
        self.idx = idx

        self.parent = None
        self.num_children = 0
        self.is_leaf = True
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.is_leaf = False
        self.children.append(child)

    def size(self):
        """
        Returns Size of the (sub-)tree rooted from this node
        """
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __str__(self):
        return self.get_ascii()

    def __vis__(self, attributes=None):
        return self.get_ascii(attributes=attributes)

    def __repr__(self):
        return self.word

    def _asciiArt(self, char1='-', show_internal=True, compact=False, attributes=None):
        """
        Returns the ASCII representation of the tree.
        Code based on the PyCogent GPL project.
        """
        if not attributes:
            attributes = ["word"]
        node_name = ', '.join(map(str, [getattr(self, v) for v in attributes if hasattr(self, v)]))

        LEN = max(3, len(node_name) if not self.children or show_internal else 3)
        PAD = ' ' * LEN
        PA = ' ' * (LEN - 1)
        if not self.is_leaf:
            mids = []
            result = []
            for c in self.children:
                if len(self.children) == 1:
                    char2 = '-'
                elif c is self.children[0]:
                    char2 = '/'
                elif c is self.children[-1]:
                    char2 = '\\'
                else:
                    char2 = '-'
                (clines, mid) = c._asciiArt(char2, show_internal, compact, attributes)
                mids.append(mid + len(result))
                result.extend(clines)
                if not compact:
                    result.append('')
            if not compact:
                result.pop()
            (lo, hi, end) = (mids[0], mids[-1], len(result))
            prefixes = [PAD] * (lo + 1) + [PA + '|'] * (hi - lo - 1) + [PAD] * (end - hi)
            mid = int((lo + hi) / 2)
            prefixes[mid] = char1 + '-' * (LEN - 1) + prefixes[mid][-1]
            result = [p + l for (p, l) in zip(prefixes, result)]
            if show_internal:
                stem = result[mid]
                result[mid] = stem[0] + node_name + stem[len(node_name) + 1:]
            return (result, mid)
        else:
            return ([char1 + '-' + node_name], 0)

    def get_ascii(self, show_internal=True, compact=False, attributes=None):
        """
        Returns a string containing an ascii drawing of the tree.
        :argument show_internal: includes internal edge names.
        :argument compact: use exactly one line per tip.
        :param attributes: A list of node attributes to shown in the
            ASCII representation.
        """
        (lines, mid) = self._asciiArt(show_internal=show_internal,
                                      compact=compact, attributes=attributes)
        return '\n' + '\n'.join(lines)


def build_bitree(tree):
    def traversal(node):
        a = node.items()
        (wd, child), = a
        node = BiTree(wd)
        for c in child:
            node.add_child(traversal(c))
        return node

    node = traversal(tree)
    return node
