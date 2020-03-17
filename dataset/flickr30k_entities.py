import base64
import csv
import pickle
import re
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import List, Dict
from xml.etree.ElementTree import parse

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

import util.logging as logging
from util.utils import normalize_bboxes, detectGT

logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)


def extract_flickr30k(split: str, img_feature_files: List[str], path: Path):
    """
    :param split: ['train', 'val', 'test']
    :param img_feature_files: list of tsvs
    :param path: path of flickr30k dataset
    :return: processed data of given split
    Hierarchy of HDF5 file:
    {
      'pos_boxes': num_images x 2           # bbox start offset, bbox end offset
      'image_bb': num_boxes x 4             # x1, y1, x2, y2
      'spatial_features': num_boxes x 6     # scaled x1, y1, x2, y2, w, h
      'image_features': num_boxes x 2048
    }
    """
    splits = ['train', 'val', 'test']
    assert split in splits, f'Unknown split {split}'

    feature_length = 2048
    known_num_boxes = {'train': 904930, 'val': 29906, 'test': 30034}

    ids_files = {split: (path / split).with_suffix('.txt') for split in splits}
    data_files = {split: (path / split).with_suffix('.hdf5') for split in splits}
    indices_files = {split: path / (split + '_imgid2idx.pkl') for split in splits}

    num_boxes = known_num_boxes[split]
    image_ids = {int(line) for line in ids_files[split].open()}
    logger.info(f'{split} num_boxes={num_boxes}')

    data_file = h5py.File(data_files[split], 'w')
    img_bb = data_file.create_dataset('image_bb', (num_boxes, 4), 'f')
    pos_boxes = data_file.create_dataset('pos_boxes', (len(image_ids), 2), dtype='int32')
    img_features = data_file.create_dataset('image_features', (num_boxes, feature_length), 'f')
    img_spatial_features = data_file.create_dataset('spatial_features', (num_boxes, 6), 'f')

    # image id => index to pos_boxes
    indices = {}
    counter = 0
    num_boxes = 0
    FIELD_NAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    for img_feature_file in img_feature_files:
        unknown_ids = []
        logger.info(f'processing {img_feature_file} ...')
        with open(img_feature_file, 'r+') as feature_file:
            reader = csv.DictReader(feature_file, delimiter='\t', fieldnames=FIELD_NAMES)
            for img_RoIs in reader:
                img_RoIs['num_boxes'] = int(img_RoIs['num_boxes'])
                img_RoIs['boxes'] = bytes(img_RoIs['boxes'], 'utf')
                img_RoIs['features'] = bytes(img_RoIs['features'], 'utf')
                image_id = int(img_RoIs['image_id'])
                image_w = float(img_RoIs['image_w'])
                image_h = float(img_RoIs['image_h'])
                bboxes = np.frombuffer(
                    base64.decodebytes(img_RoIs['boxes']), dtype=np.float32
                ).reshape((img_RoIs['num_boxes'], 4))

                # bbox: x1, y1, x2, y2
                # spatial_feature: scaled(x1, y1, x2, y2, w, h)
                spatial_features = normalize_bboxes(bboxes, image_w, image_h)

                if image_id in image_ids:
                    image_ids.remove(image_id)
                    indices[image_id] = counter  # index to pos_boxes
                    box_idx_st = num_boxes
                    box_idx_ed = box_idx_st + img_RoIs['num_boxes']
                    pos_boxes[counter, :] = np.array([box_idx_st, box_idx_ed])
                    img_bb[box_idx_st: box_idx_ed, :] = bboxes
                    img_spatial_features[box_idx_st: box_idx_ed, :] = spatial_features
                    img_features[box_idx_st: box_idx_ed, :] = np.frombuffer(
                        base64.decodebytes(img_RoIs['features']), dtype=np.float32
                    ).reshape((img_RoIs['num_boxes'], -1))
                    counter += 1
                    num_boxes += img_RoIs['num_boxes']
                else:
                    # out of split
                    unknown_ids.append(image_id)

        logger.info(f'{len(unknown_ids)} out of {split} split ids...')
        logger.info(f'{len(image_ids)} image_ids left...')

    if len(image_ids) != 0:
        logger.warning('Warning: %s_image_ids is not empty' % split)

    pickle.dump(indices, open(indices_files[split], 'wb'))
    data_file.close()
    logger.info(f'Saved {split} features to {data_files[split]}')
    return data_file


# def _load_flickr30k_tree(split: str,
#                          path: str,
#                          imgid2idx: Dict[int, int],
#                          bbox_offsets: torch.Tensor,
#                          rois: torch.Tensor):
#     path = Path(path)
#     cache = path / f'{split}_entities_tree.pt'
#     cache1 = path / f'{split}_entities_tree1.pt'
#     if cache.exists():
#         logger.info(f'Loading entities from cache at {cache}')
#         return torch.load(cache)
#
#     logger.info(f'Extracting entities from scratch...')
#     pattern_no = r'\/EN\#(\d+)'
#     pattern_phrase = r'\[(.*?)\]'
#     pattern_annotation = r'\[[^ ]+ '
#
#     num_grounding = OrderedDict()
#     num_captions = OrderedDict()
#     missing_entity_count = defaultdict(int)
#     multibox_entity_count = 0
#     img_cap_entries = []
#
#     nlp = spacy.load('en_core_web_sm', disable=['ner'])
#     for imgid, img_idx in tqdm(imgid2idx.items(), desc=f'Loading Flickr30K'):
#         phrase_file = path / f'Sentences/{imgid}.txt'  # entity coreference chain
#         anno_file = path / f'Annotations/{imgid}.xml'  # entity detected_RoIs
#         with open(phrase_file, 'r', encoding='utf-8') as f:
#             sentences = [line.strip() for line in f]
#
#         # Parse Annotation to retrieve GT detected_RoIs for each entity id
#         #   GT box => one or more entity ids
#         #   entity id => one or more GT boxes
#         root = parse(anno_file).getroot()
#         objects = root.findall('./object')
#         entity_GT_boxes = defaultdict(list)
#         for obj in objects:
#             # Exceptions: too many, scene or non-visual
#             if obj.find('bndbox') is None or len(obj.find('bndbox')) == 0:
#                 continue
#
#             x1 = int(obj.findtext('./bndbox/xmin'))
#             y1 = int(obj.findtext('./bndbox/ymin'))
#             x2 = int(obj.findtext('./bndbox/xmax'))
#             y2 = int(obj.findtext('./bndbox/ymax'))
#             assert x1 > 0 and y1 > 0
#             for name in obj.findall('name'):
#                 entity_ID = int(name.text)
#                 assert entity_ID > 0
#
#                 if entity_ID in entity_GT_boxes:
#                     multibox_entity_count += 1
#                 entity_GT_boxes[entity_ID].append([x1, y1, x2, y2])
#
#         # Parse Sentence: caption and phrases w/ grounding
#         #   entity id => one or more phrases
#         start, end = bbox_offsets[img_idx]
#         detected_RoIs = rois[start: end]
#         num_captions[imgid] = len(sentences)
#         num_grounding[imgid] = 0
#
#         for sent_id, sent in enumerate(sentences):
#
#             caption = re.sub(pattern_annotation, '', sent).replace(']', '')
#             tree_sent = caption.replace(', ', '').replace('.', '')
#
#             doc = nlp(tree_sent)
#             new_sent, new, old_sent, old = doc_prune(doc)
#             if len(new) == 0:
#                 print("we must examine the dataset!")
#                 print(sentences[sent_id])
#                 raise KeyError
#             doc = nlp(new_sent)
#             t, s = doc_to_tree(doc)
#             s = sorted(s, key=lambda x: x[0])
#             token = [x[1] for x in s]
#             tag = [x[3] for x in s]
#             dep = [x[4] for x in s]
#             tree = {'tree': t, 'tokens': token,
#                     'tags': tag, 'deps': dep}
#
#             entities = []
#             for i, entity in enumerate(re.findall(pattern_phrase, sent)):
#                 info, phrase = entity.split(' ', 1)
#                 types = info.split('/')[2:]
#                 entity_ID = int(re.findall(pattern_no, info)[0])
#
#                 # grounded RoIs
#                 if entity_ID not in entity_GT_boxes:
#                     assert entity_ID >= 0
#                     for t in types:
#                         missing_entity_count[t] += 1
#                     continue
#
#                 # find matched ROI indices with entity GT boxes
#                 matched_RoIs = detectGT(entity_GT_boxes[entity_ID], detected_RoIs)
#                 entities.append((entity_ID, types, phrase, matched_RoIs))
#                 if not matched_RoIs:
#                     logger.warning(f'No object detection of GT: [{imgid}][{i}:{entity_ID}]{phrase}')
#
#             if not entities:
#                 logger.warning(f'[{imgid}] no entity RoIs found: {sent}')
#                 continue
#
#             num_grounding[imgid] += 1
#             img_cap_entries.append({
#                 'imgid': imgid,
#                 'img_idx': img_idx,
#                 'caption': caption,
#                 'entities': entities,
#                 'tree': tree
#             })
#
#     if len(missing_entity_count) > 0:
#         cap = torch.tensor(list(num_captions.values()))
#         grounded = torch.tensor(list(num_grounding.values()))
#         incomplete = (grounded < cap).sum().item()
#         none = (grounded == 0).sum().item()
#         if none > 0:
#             indexes = (grounded == 0).nonzero().view(-1)
#             imgids = list(num_grounding.keys())
#             imgids = tuple(imgids[i] for i in indexes)
#             logger.warning(f"images w/o entity grounding: {imgids}")
#
#         logger.warning(
#             f"{incomplete}/{len(num_grounding)} with incomplete caption num_grounding"
#             f", {none}/{incomplete} w/o num_grounding"
#         )
#         logger.warning(
#             f"missing_entity_count: {', '.join(f'{k}={v}' for k, v in missing_entity_count.items())}")
#         logger.warning(f"multibox_entity_count={multibox_entity_count}")
#
#     torch.save(img_cap_entries, cache)
#
#     trees = []
#     for entry in img_cap_entries:
#         trees.append(entry['tree'])
#     word_vocab = build_vocab(trees, 'tokens', 1)
#     tag_vocab = build_vocab(trees, 'tags', 1)
#     dep_vocab = build_vocab(trees, 'deps', 1)
#     wtoi = {w: i for i, w in enumerate(word_vocab)}
#     ttoi = {w: i for i, w in enumerate(tag_vocab)}
#     dtoi = {w: i for i, w in enumerate(dep_vocab)}
#     info = {
#         'word_to_ix': wtoi,
#         'tag_to_ix': ttoi,
#         'dep_to_ix': dtoi
#     }
#     torch.save(info, cache1)
#     return img_cap_entries, info
#

def _load_flickr30k(split: str,
                    path: str,
                    imgid2idx: Dict[int, int],
                    bbox_offsets: torch.Tensor,
                    rois: torch.Tensor):
    """
    Load img_cap_entries of entity detected_RoIs by ids.

    :param split: ['train', 'val', 'test']
    :param path: saved path to Flickr30K annotation dataset
    :param imgid2idx:  dict {image_id -> offset to ROI features/detected_RoIs}
    :param bbox_offsets: start and end indexes of bounding boxes of certain img-idx
    :param rois: detected RoIs
    :return: list of entries, every entry is
            {
                'imgid': imgid,
                'img_idx': img_idx,
                'caption': caption,
                'entities': entities,
            }
    """

    path = Path(path)
    cache = path / f'{split}_entities.pt'
    if cache.exists():
        logger.info(f'Loading entities from cache at {cache}')
        return torch.load(cache)

    logger.info(f'Extracting entities from scratch...')
    pattern_no = r'\/EN\#(\d+)'
    pattern_phrase = r'\[(.*?)\]'
    pattern_annotation = r'\[[^ ]+ '

    num_grounding = OrderedDict()
    num_captions = OrderedDict()
    missing_entity_count = defaultdict(int)
    multibox_entity_count = 0
    img_cap_entries = []
    for imgid, img_idx in tqdm(imgid2idx.items(), desc=f'Loading Flickr30K'):
        phrase_file = path / f'Sentences/{imgid}.txt'  # entity coreference chain
        anno_file = path / f'Annotations/{imgid}.xml'  # entity detected_RoIs
        with open(phrase_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f]

        # Parse Annotation to retrieve GT detected_RoIs for each entity id
        #   GT box => one or more entity ids
        #   entity id => one or more GT boxes
        root = parse(anno_file).getroot()
        objects = root.findall('./object')
        entity_GT_boxes = defaultdict(list)
        for obj in objects:
            # Exceptions: too many, scene or non-visual
            if obj.find('bndbox') is None or len(obj.find('bndbox')) == 0:
                continue

            x1 = int(obj.findtext('./bndbox/xmin'))
            y1 = int(obj.findtext('./bndbox/ymin'))
            x2 = int(obj.findtext('./bndbox/xmax'))
            y2 = int(obj.findtext('./bndbox/ymax'))
            assert x1 > 0 and y1 > 0
            for name in obj.findall('name'):
                entity_ID = int(name.text)
                assert entity_ID > 0

                if entity_ID in entity_GT_boxes:
                    multibox_entity_count += 1
                entity_GT_boxes[entity_ID].append([x1, y1, x2, y2])

        # Parse Sentence: caption and phrases w/ grounding
        #   entity id => one or more phrases
        start, end = bbox_offsets[img_idx]
        detected_RoIs = rois[start: end]
        num_captions[imgid] = len(sentences)
        num_grounding[imgid] = 0
        for sent_id, sent in enumerate(sentences):
            caption = re.sub(pattern_annotation, '', sent).replace(']', '')

            entities = []
            for i, entity in enumerate(re.findall(pattern_phrase, sent)):
                info, phrase = entity.split(' ', 1)
                types = info.split('/')[2:]
                entity_ID = int(re.findall(pattern_no, info)[0])

                # grounded RoIs
                if entity_ID not in entity_GT_boxes:
                    assert entity_ID >= 0
                    for t in types:
                        missing_entity_count[t] += 1
                    continue

                # find matched ROI indices with entity GT boxes
                matched_RoIs = detectGT(entity_GT_boxes[entity_ID], detected_RoIs)
                entities.append((entity_ID, types, phrase, matched_RoIs))
                if not matched_RoIs:
                    logger.warning(f'No object detection of GT: [{imgid}][{i}:{entity_ID}]{phrase}')

            if not entities:
                logger.warning(f'[{imgid}] no entity RoIs found: {sent}')
                continue

            num_grounding[imgid] += 1
            img_cap_entries.append({
                'imgid': imgid,
                'img_idx': img_idx,
                'caption': caption,
                'entities': entities,
            })

    if len(missing_entity_count) > 0:
        cap = torch.tensor(list(num_captions.values()))
        grounded = torch.tensor(list(num_grounding.values()))
        incomplete = (grounded < cap).sum().item()
        none = (grounded == 0).sum().item()
        if none > 0:
            indexes = (grounded == 0).nonzero().view(-1)
            imgids = list(num_grounding.keys())
            imgids = tuple(imgids[i] for i in indexes)
            logger.warning(f"images w/o entity grounding: {imgids}")

        logger.warning(
            f"{incomplete}/{len(num_grounding)} with incomplete caption num_grounding"
            f", {none}/{incomplete} w/o num_grounding"
        )
        logger.warning(
            f"missing_entity_count: {', '.join(f'{k}={v}' for k, v in missing_entity_count.items())}")
        logger.warning(f"multibox_entity_count={multibox_entity_count}")

    torch.save(img_cap_entries, cache)
    return img_cap_entries


def lastTokenIndex(caption_tokens, phrase_tokens):
    """

    Args:
        caption_tokens: list of tokens
        phrase_tokens: list of phrase tokens
    Return the index of the last token in the sublist
    """
    phrase_len = len(phrase_tokens)
    idx = -1
    while True:
        try:
            idx = caption_tokens.index(phrase_tokens[0], idx + 1)
        except ValueError:
            return -1
        if phrase_tokens == caption_tokens[idx: idx + phrase_len]:
            return idx + phrase_len - 1


class Flickr30kEntities(Dataset):
    path_config = {
        'images': 'flickr30k_images',
        'captions': 'results.tsv',
        'sentences': 'Sentences',
        'annotations': 'Annotations',
        'features': {
            'base': 'features/resnet101-faster-rcnn-vg-100-2048',
            'train': [
                'train_flickr30k_resnet101_faster_rcnn_genome.tsv.1',
                'train_flickr30k_resnet101_faster_rcnn_genome.tsv.2',
            ],
            'val': ['val_flickr30k_resnet101_faster_rcnn_genome.tsv.3'],
            'test': ['test_flickr30k_resnet101_faster_rcnn_genome.tsv.3'],
        },
    }

    EType2Id = dict(people=0, clothing=1, bodyparts=2, animals=3,
                    vehicles=4, instruments=5, scene=6, other=7)
    ETypes = list(EType2Id.keys())

    def __init__(
            self,
            split: str,
            path: str,
            tokenization,
            max_tokens=80,
            max_entities=16,
            max_rois=100,
            training=True):

        path = Path(path)
        data_file = path / f'{split}.hdf5'
        imgid2idx = path / f'{split}_imgid2idx.pkl'
        if not data_file.exists() or not imgid2idx.exists():
            logging.warning(f'{data_file} or {imgid2idx} not exist, extracting features on the fly...')
            prefix = path / self.path_config['features']['base']
            tsvs = [prefix / tsv for tsv in self.path_config['features'][split]]
            logger.info(f'Extracting ROI features from {prefix}')
            extract_flickr30k(split, tsvs, path)

        logger.info(f'Loading image/RoI features from {data_file}')
        self.imgid2idx = pickle.load(open(imgid2idx, 'rb'))
        with h5py.File(data_file, 'r') as data_file:
            self.offsets = torch.from_numpy(np.array(data_file.get('pos_boxes')))
            self.features = torch.from_numpy(np.array(data_file.get('image_features')))
            self.spatials = torch.from_numpy(np.array(data_file.get('spatial_features')))
            self.bboxes = torch.from_numpy(np.array(data_file.get('image_bb')))

        self.max_tokens = max_tokens
        self.max_entities = max_entities
        self.max_rois = max_rois
        self.tokenization = tokenization
        self.img_cap_entries = _load_flickr30k(split, path, self.imgid2idx, self.offsets, self.bboxes)
        self.training = training
        if tokenization in ['bert', 'wordpiece']:
            from models.nlp import bert
            bert.setup()

    def tensorize(self, entry):
        if self.tokenization in ['bert', 'wordpiece']:
            from models.nlp import bert
            entities = entry['entities']
            tokens = bert.tokenize(entry['caption'], plain=False)
            indices = -torch.ones(self.max_entities, dtype=torch.long)  # padded with -1 up to max_entities
            target = torch.zeros(self.max_entities, self.max_rois)  # padded up to (max_entities x max_rois)
            eTypes = torch.zeros(self.max_entities, len(Flickr30kEntities.ETypes))
            for i, entity in enumerate(entities):
                entity_ID, types, phrase, rois = entity
                toks = bert.tokenize(phrase, plain=False)[1:-1]
                last_token_idx = lastTokenIndex(tokens, toks)
                assert last_token_idx >= 0, f'Cannot locate phrase[{i}]={toks}'
                if last_token_idx < self.max_tokens:
                    indices[i] = last_token_idx
                    target[i][rois] = 1
                    # logger.info(f'phrase[{i}:{entity_ID}]={toks}, last_index={last_token_idx}, rois={rois}')
                else:
                    logger.warning(
                        f'Truncated phrase: "{phrase}", last token index {last_token_idx} >= {self.max_tokens}')
                eTypes[i][list(Flickr30kEntities.EType2Id[t] for t in types)] = 1

            # BERT ids, mask and segment by truncating or padding input tokens up to max_tokens
            token_ids, token_seg, token_mask = bert.tensorize(tokens, max_tokens=self.max_tokens)
            return (token_ids, token_seg, token_mask), indices, target, eTypes
        else:
            # LSTM
            raise NotImplementedError('tokenization for LSTM is not implemented yet')

    def __getitem__(self, index):
        entry = self.img_cap_entries[index]
        imgidx = entry['img_idx']
        start, end = self.offsets[imgidx]
        rois = (end - start).item()
        features = self.features[start:end, :]
        spatials = self.spatials[start:end, :]
        features = F.pad(features, [0, 0, 0, self.max_rois - rois])
        spatials = F.pad(spatials, [0, 0, 0, self.max_rois - rois])
        mask = torch.tensor([1] * rois + [0] * (self.max_rois - rois))
        tokens, indices, target, types = self.tensorize(entry)

        if self.training:
            return (features, spatials, mask, *tokens), (indices, target, types)  # (x), (y)
        else:
            return (features, spatials, mask, *tokens), (indices, target, types), entry

    def __len__(self):
        return len(self.img_cap_entries)

    def validate(self):
        rois_max = 0
        tokens_max = 0
        entities_max = 0
        for i, entry in enumerate(self.img_cap_entries):
            (features, spatials, mask, *tokens), (indices, target, types) = self[i]
            rois_max = max(rois_max, features.shape[0])
            entities_max = max(entities_max, (indices <= 0).nonzero()[0].item())
            if self.tokenization in ['bert', 'wordpiece']:
                tokens_max = max(tokens_max, tokens[2].sum().item())

        assert rois_max <= self.max_rois
        assert tokens_max <= self.max_tokens
        assert entities_max <= self.max_entities
        logger.info(f'max (rois, tokens, entities) = ({rois_max}, {tokens_max}, {entities_max})')


if __name__ == '__main__':
    logging.basicConfig()
    flickr30k_dir = Path('./data/flickr30k_entities')
    dataset = Flickr30kEntities('test', flickr30k_dir, 'bert')
    dataset.validate()
