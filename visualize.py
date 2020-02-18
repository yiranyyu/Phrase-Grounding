import torch
from pathlib import Path
from torch.utils.data import DataLoader

from dataset.flickr30k_entities import Flickr30kEntities, lastTokenIndex
from util import logging
from models.bert import IBertConfig, BertForGrounding, select
import copy
import cv2
from models.nlp import bert

RED = (0, 0, 255)


def img_read_if_str(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    return img


def draw_rectangle(img, bbox, color=RED, thickness=2):
    img = img_read_if_str(img)
    if isinstance(bbox, dict):
        bbox = [
            bbox['x1'],
            bbox['y1'],
            bbox['x2'],
            bbox['y2'],
        ]
    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[0] = min(bbox[0], img.shape[1])
    bbox[1] = min(bbox[1], img.shape[0])
    bbox[2] = max(bbox[2], 0)
    bbox[3] = max(bbox[3], 0)
    bbox[2] = min(bbox[2], img.shape[1])
    bbox[3] = min(bbox[3], img.shape[0])
    assert bbox[2] >= bbox[0]
    assert bbox[3] >= bbox[1]
    assert bbox[0] >= 0
    assert bbox[1] >= 0
    assert bbox[2] <= img.shape[1]
    assert bbox[3] <= img.shape[0]
    cur_img = copy.deepcopy(img)
    cv2.rectangle(
        cur_img,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        color,
        thickness)
    return cur_img


def gen_caption_str(caption_tokens, phrase_tokens):
    ed_idx = lastTokenIndex(caption_tokens, phrase_tokens) + 1
    st_idx = ed_idx - len(phrase_tokens)
    caption = caption_tokens[:st_idx] + ['['] + phrase_tokens + [']'] + caption_tokens[ed_idx:]
    return ' '.join(caption)


def visualize_contextual_grounding_flickr30k(model,
                                             batch,
                                             output_prefix: Path,
                                             path: Path,
                                             dataset: Flickr30kEntities,
                                             interactive=False):
    """
    Visualize a Caption-Image pair
    entry structure:
            {
            'imgid': imgid,
            'img_idx': img_idx,
            'caption': caption,
            'entities': entities,
            }
    x, y, entry = batch
    features, spatials, mask, token_ids, token_segs, token_mask = x
    indices, target, types = y

    :param output_prefix:
    :param dataset:
    :param interactive:
    :param model: pre-trained model
    :param batch: input
    :param path: base path of flickr30k entities dataset
    """
    if not output_prefix.exists():
        output_prefix.mkdir(parents=True)

    img_prefix = path / 'flickr30k_images'
    x, y, entry = batch
    assert x[0].shape[0] == 1, f'batch size must be 1 to avoid entity lose, but got {len(batch)}'
    indices, *_ = y
    output = model(x).unsqueeze(0)
    output, target, num_entities, types = select(output, y)
    imgids, img_idxes, captions, entities = entry['imgid'], entry['img_idx'], entry['caption'], entry['entities']
    assert len(entities) == num_entities
    k = 5

    imgid = imgids[0]
    img_idx = img_idxes[0]
    caption_tokens = bert.tokenize(captions[0], plain=True)
    img_name = Path(str(imgid.item())).with_suffix('.jpg')
    img_path = img_prefix / img_name
    origin_img = cv2.imread(str(img_path))

    start, end = dataset.offsets[img_idx]
    detected_RoIs = dataset.bboxes[start:end]

    for entity_idx, (logits, entity) in enumerate(zip(output, entities)):
        softmax = logits.softmax(dim=0)
        probs, predicted_RoIs = torch.topk(softmax, k=k)

        entity_ID, types, phrase, matched_RoIs = entity
        entity_ID = entity_ID[0].item()
        types = types[0]
        phrase = phrase[0]
        phrase_tokens = bert.tokenize(phrase, plain=True)
        caption = gen_caption_str(caption_tokens, phrase_tokens)

        matched_RoI_indexes = set(x.item() for x in matched_RoIs)
        if not matched_RoI_indexes:
            continue
        matched_RoIs = [detected_RoIs[x] for x in matched_RoI_indexes]

        topk_RoI_idx = set(x.item() for x in predicted_RoIs)
        topk_RoIs = [detected_RoIs[x] for x in topk_RoI_idx]
        if not topk_RoI_idx.intersection(matched_RoI_indexes):
            target_img = origin_img
            for RoI in matched_RoIs:
                target_img = draw_rectangle(target_img, RoI)

            topk_img = target_img
            for idx, RoI in enumerate(topk_RoIs):
                color = (214, 172, 29)
                topk_img = draw_rectangle(topk_img, RoI, color=color)
                margin = 5
                topk_img = cv2.putText(topk_img, str(idx + 1), (RoI[0] + margin, max(10, RoI[3] - margin)),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                       color=RED, thickness=2)
            output_name = '%s_%s_%s' % (imgid.item(), entity_ID, caption)
            output_path = (output_prefix / output_name).with_suffix('.jpg')
            cv2.imwrite(str(output_path), topk_img)

            if interactive:
                win_name = '(%s, %s) [%s] %s' % (imgid.item(), entity_ID, types, phrase)
                cv2.imshow(win_name, target_img)
                cv2.waitKey()
                cv2.destroyWindow(win_name)


if __name__ == '__main__':
    output_dir = Path('./export')
    model_prefix = Path('bert-adam-s1204-L1-H2-dp0.4-b256-lr0.00005-wp0.1-abs')
    output_prefix = output_dir / model_prefix

    model_name = 'grounding_model_4_recall=0.7152131.pth'
    cfgI = IBertConfig(
        hidden_size=2048,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=3072,
        hidden_dropout_prob=0.4,
        attention_probs_dropout_prob=0.4,
        spatial='abs',
    )
    logging.info('Initializing model ...')
    model = BertForGrounding(cfgI).eval()
    model.load(output_prefix / 'model' / model_name, map_location=torch.device('cpu'))
    logging.info('Model loaded.')

    logging.info('Loading dataset ...')
    flickr30k_dir = Path('./data/flickr30k_entities')
    dataset = Flickr30kEntities('test', flickr30k_dir, 'bert', training=False)
    logging.info('Dataset loaded.')

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # bs must be 1 to avoid entity trunc
    for batch in data_loader:
        visualize_contextual_grounding_flickr30k(model,
                                                 batch,
                                                 output_prefix / 'img',
                                                 path=flickr30k_dir,
                                                 dataset=dataset)
