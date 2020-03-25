__author__ = 'licheng'  # modified by yirany

"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import csv
import sys
import json
import time
import pickle
import base64
import itertools
import os.path as osp
import numpy as np
import h5py
import torch
from typing import Dict, Any
from pathlib import Path

from torch.utils.data import Dataset
from typing import List
from tqdm import tqdm
import torch.nn.functional as F

from util.utils import normalize_bboxes, detectGT
from util import logging
from models.nlp import bert
from dataset import FIELD_NAMES, _count_bbox_num

csv.field_size_limit(sys.maxsize)
bert.setup()


# author: 'licheng'  # modified by yirany
class REFER:
    def __init__(self, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)

        data_root = Path(__file__).resolve().parent.parent / 'data/referit'
        self.DATA_DIR = data_root / dataset
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        else:
            print('No refer dataset is called [%s]' % dataset)
            exit()

        # Load refs from referit/{dataset}/refs(dataset).json
        start_time = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        self.data: Dict[str, Any] = {'dataset': dataset, 'refs': pickle.load(open(ref_file, 'rb'))}

        # Load annotations from referit/{dataset}/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')

        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id: int = ref['ref_id']
            ann_id: str = ref['ann_id']
            category_id: int = ref['category_id']
            image_id: int = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')
        print('DONE (t=%.2fs)' % (time.time() - start_time))

    def getRefIds(self, image_ids=None, cat_ids=None, ref_ids=None, split=''):
        if ref_ids is None:
            ref_ids = []
        if cat_ids is None:
            cat_ids = []
        if image_ids is None:
            image_ids = []

        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if image_ids or cat_ids or ref_ids or split:
            if image_ids:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if cat_ids:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if ref_ids:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]

            if split:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    exit()
        else:
            refs = self.data['refs']

        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=None, cat_ids=None, ref_ids=None):
        if ref_ids is None:
            ref_ids = []
        if cat_ids is None:
            cat_ids = []
        if image_ids is None:
            image_ids = []

        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if
                         image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ann_ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=None):
        if ref_ids is None:
            ref_ids = []
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=None):
        ref_ids = [] if ref_ids is None else ref_ids

        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=None):
        ann_ids = [] if ann_ids is None else ann_ids

        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == str:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=None):
        image_ids = [] if image_ids is None else image_ids

        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=None):
        cat_ids = [] if cat_ids is None else cat_ids

        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]


# noinspection PyShadowingNames
def extract_referit_game(split: str, file_name: str, data_path: Path):
    splits = ['train', 'val', 'test']
    assert split in splits, f'Unknown split {split}'

    feature_length = 2048
    file_name = data_path / file_name

    ids_files = {split: (data_path / f'{split}_imgids.txt') for split in splits}
    data_files = {split: (data_path / split).with_suffix('.hdf5') for split in splits}
    indices_files = {split: data_path / (split + '_imgid2idx.pkl') for split in splits}

    split_img_ids = set(json.load(ids_files[split].open()))
    num_boxes = _count_bbox_num(split_img_ids, file_name)
    logging.info(f'{split} num_boxes={num_boxes}')

    data_file = h5py.File(data_files[split], 'w')
    img_bb = data_file.create_dataset('image_bb', (num_boxes, 4), 'f')
    pos_boxes = data_file.create_dataset('pos_boxes', (len(split_img_ids), 2), dtype='int32')
    img_features = data_file.create_dataset('image_features', (num_boxes, feature_length), 'f')
    img_spatial_features = data_file.create_dataset('spatial_features', (num_boxes, 6), 'f')

    # image id => index to pos_boxes
    indices = {}
    num_boxes = 0
    collected_imgids = set()
    with file_name.open() as img_feature_file:
        reader = csv.DictReader(img_feature_file, delimiter='\t', fieldnames=FIELD_NAMES)
        for img_data in tqdm(reader, desc=f'Extracting {split}'):
            img_id = int(img_data['image_id'])
            n_boxes = int(img_data['num_boxes'])
            image_w = float(img_data['image_w'])
            image_h = float(img_data['image_h'])

            bboxes = eval(bytes(img_data['boxes'], 'utf8'))
            features = eval(bytes(img_data['features'], 'utf8'))
            bboxes = np.frombuffer(base64.decodebytes(bboxes), dtype=np.float64).reshape(n_boxes, 4)
            features = np.frombuffer(base64.decodebytes(features), dtype=np.float32).reshape(n_boxes, feature_length)
            spatial_features = normalize_bboxes(bboxes, image_w, image_h)

            if img_id in split_img_ids:
                if img_id in collected_imgids:
                    logging.warning(f'Duplicated img feature instance of image {img_id}')

                n_collected = len(collected_imgids)
                indices[img_id] = n_collected  # index to pos_boxes
                box_idx_st = num_boxes
                box_idx_ed = box_idx_st + n_boxes
                pos_boxes[n_collected, :] = np.array([box_idx_st, box_idx_ed])
                img_bb[box_idx_st: box_idx_ed, :] = bboxes
                img_spatial_features[box_idx_st: box_idx_ed, :] = spatial_features
                img_features[box_idx_st: box_idx_ed, :] = features

                num_boxes = box_idx_ed
                collected_imgids.add(img_id)

    not_found = split_img_ids - collected_imgids
    if not_found:
        logging.warning(f'Warning: feature of image {not_found} in {split} not found')

    pickle.dump(indices, open(indices_files[split], 'wb'))
    data_file.close()
    logging.info(f'Saved {split} features to {data_files[split]}')
    return data_file


# noinspection PyShadowingNames
def _load_referit_game(split: str,
                       data_dir: Path,
                       split_refs: List[dict],
                       anns: Dict[str, dict],
                       imgid2idx: Dict[int, int],
                       bbox_offsets: torch.Tensor,
                       rois: torch.Tensor):
    cache = data_dir / f'{split}_cache.pt'
    if cache.exists():
        logging.info(f'Loading ref-sent pairs from cache at {cache}')
        return torch.load(cache)

    entries = []
    lost_ref = 0
    for ref in tqdm(split_refs, desc=f'Loading referIt Game {split}'):
        ann = anns[ref['ann_id']]
        GT_box = ann['bbox']
        GT_box = [GT_box[0], GT_box[1], GT_box[0] + GT_box[2], GT_box[1] + GT_box[3]]
        category = ann['category_id']
        image_id = ann['image_id']
        sentences = ref['sentences']

        img_idx = imgid2idx[image_id]
        st, ed = bbox_offsets[img_idx]
        detected_RoIs = rois[st:ed]

        matched_RoI_ids = detectGT([GT_box], detected_RoIs)
        if not matched_RoI_ids:
            # logging.warning(f'No object detection of GT: [{image_id}][{ref["ref_id"]}]')
            lost_ref += 1

        for sent_id, sent in enumerate(sentences):
            sent = sent['sent']
            entries.append({
                'imgid': image_id,
                'img_idx': img_idx,
                'sent': sent,
                'category': category,
                'matched_RoI_ids': matched_RoI_ids,
                'RoIs': detected_RoIs
            })
    if lost_ref:
        logging.warning(f'Missing {lost_ref}/{len(split_refs)} refs')

    torch.save(entries, cache)
    return entries


class ReferItGame(Dataset):
    """
    19_997 Images
    99_523 Annotations
    99_296 Refs

    Image: {'id': int,
            'height': int,
            'width': int,
            'file_name': str}

    Annotation: {'id': str
                 'bbox': List[int],
                 'category_id': int,
                 'image_id': int
                 'area': int,
                 'segmentation': List[dict],
                 'mask_name': str}

    Ref: {'ref_id': int,
          'ann_id': str,
          'image_id': int,
          'sent_ids': List[int],
          'category_id': int,
          'sentences': [
                    {'tokens': List[str],
                     'raw': str,
                     'sent_id': int,
                     'sent': str}
                ]
          }

    No overlap:
        Train:
            8_998 images
            44_681 refs
        Val:
            1_000 images
            4_865 refs
        Test:
            9_999 images
            49_750 refs


    """
    path_config = {
        'features': 'referItGame-ResNet101-pretrained-on-VG_th=0.1.tsv'
    }
    EType2Id = dict()
    ETypes = list()

    # noinspection PyShadowingNames
    def __init__(self,
                 split: str,
                 data_dir: Path,
                 max_rois=100,
                 max_tokens=44):
        assert split in ['train', 'val', 'test'], f'unrecognized split: {split}'
        self.data = REFER('refclef', 'berkeley')

        self.ref_ids = self.data.getRefIds(split=split)
        self.refs = self.data.loadRefs(self.ref_ids)
        self.all_anns = self.data.Anns

        data_file = data_dir / f'{split}.hdf5'
        imgid2idx = data_dir / f'{split}_imgid2idx.pkl'
        if not data_file.exists() or not imgid2idx.exists():
            logging.warning(f'{data_file} or {imgid2idx} not exist, extracting features on the fly...')
            extract_referit_game(split, self.path_config['features'], data_dir)

        logging.info(f'Loading image/RoI features from {data_file}')
        self.imgid2idx = pickle.load(imgid2idx.open('rb'))
        with h5py.File(data_file, 'r') as data_file:
            self.offsets = torch.from_numpy(np.array(data_file.get('pos_boxes')))
            self.features = torch.from_numpy(np.array(data_file.get('image_features')))
            self.spatials = torch.from_numpy(np.array(data_file.get('spatial_features')))
            self.bboxes = torch.from_numpy(np.array(data_file.get('image_bb')))

        self.entries = _load_referit_game(split, data_dir, self.refs, self.all_anns, self.imgid2idx,
                                          self.offsets, self.bboxes)
        self.max_rois = max_rois
        self.max_tokens = max_tokens
        self.max_entities = 1
        ReferItGame.EType2Id = {b: int(a) for a, b in
                                [line.strip().split('\t', 2) for line in (data_dir / 'categories.txt').open()]}
        ReferItGame.ETypes = list(ReferItGame.EType2Id.keys())

    def tensorize(self, entry):
        """
        entry:{
            'imgid': image_id,
            'img_idx': img_idx,
            'sent': sent,
            'category': category,
            'matched_RoI_ids': matched_RoI_ids,
            'RoIs': detected_RoIs
        }

        :param entry: img-boxes-sent pair
        """
        sent = entry['sent']
        tokens, piece2word = bert.tokenize(sent, plain=False)
        indices = torch.zeros(1, dtype=torch.long)
        target = torch.zeros(1, self.max_rois)
        category = torch.zeros(1, len(ReferItGame.ETypes))

        category[0][entry['category']] = 1
        target[0][entry['matched_RoI_ids']] = 1
        indices[0] = len(tokens) - 1

        # BERT ids, mask and segment by truncating or padding input tokens up to max_tokens
        token_ids, token_seg, token_mask = bert.tensorize(tokens, max_tokens=self.max_tokens)
        return (token_ids, token_seg, token_mask), indices, target, category

    def __getitem__(self, index):
        entry = self.entries[index]
        img_idx = entry['img_idx']
        st, ed = self.offsets[img_idx]
        rois = (ed - st).item()
        features = self.features[st:ed]
        spatials = self.spatials[st:ed]
        features = F.pad(features, [0, 0, 0, self.max_rois - rois])
        spatials = F.pad(spatials, [0, 0, 0, self.max_rois - rois])
        mask = torch.tensor([1] * rois + [0] * (self.max_rois - rois))
        tokens, indices, target, types = self.tensorize(entry)
        return (features, spatials, mask, *tokens), (indices, target, types)

    def __len__(self):
        return len(self.entries)

    def validate(self):
        rois_max = 0
        tokens_max = 0
        for i, entry in enumerate(self.entries):
            (features, spatials, mask, *tokens), (indices, target, types) = self[i]
            rois_max = max(rois_max, features.shape[0])
            tokens_max = max(tokens_max, tokens[2].sum().item())

        assert rois_max <= self.max_rois
        assert tokens_max <= self.max_tokens
        logging.info(f'max (rois, tokens) = ({rois_max}, {tokens_max})')


if __name__ == '__main__':
    logging.basicConfig()
    splits = ['train', 'val', 'test']
    data_dir = Path('/home/yty/workspace/ACM2020/data/referit/refclef')
    for split in splits:
        data_set = ReferItGame(split, data_dir)
        print(f'{split}: {len(data_set)} entries, {len(data_set.refs)} refs')
