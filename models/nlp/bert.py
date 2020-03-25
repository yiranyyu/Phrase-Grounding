import os
from pathlib import Path

import pytorch_pretrained_bert as BERT
import torch
from pytorch_pretrained_bert import (
    BertTokenizer,
)
from pytorch_pretrained_bert.modeling import (
    WEIGHTS_NAME,
    CONFIG_NAME,  # 0.6.1
)
from pytorch_pretrained_bert.tokenization import VOCAB_NAME

from util import logging

_PRETRAINED = None
_TOKENIZER = None
SEP = None
PAD = None

logger = logging.getLogger(__name__)


def setup(base=True, uncased=True):
    global _PRETRAINED
    global _TOKENIZER
    global SEP
    global PAD
    _PRETRAINED = "bert-{}-{}".format(
        base and "base" or "large", uncased and "uncased" or "cased"
    )
    _TOKENIZER = BertTokenizer.from_pretrained(_PRETRAINED)
    SEP = stoi("[SEP]")
    PAD = stoi("[PAD]")


def pretrained():
    return _PRETRAINED


def itos(ids):
    if isinstance(ids, list):
        return _TOKENIZER.ids_to_tokens(ids)
    else:
        return _TOKENIZER.ids_to_tokens([ids])[0]


def stoi(tokens):
    if isinstance(tokens, list):
        return _TOKENIZER.convert_tokens_to_ids(tokens)
    else:
        return _TOKENIZER.vocab[tokens]


def tokenize(text, plain=True, max_tokens=None):
    """Tokenize text up to a fixed length.

    Args:
        text: a sentence of a list of sentences
        plain: whether to insert BERT tokens
        max_tokens: max number of tokens if given
    """

    assert (
            _TOKENIZER is not None
    ), "setup(base, uncased) must be called to initialize the BERT tokenizer first"

    text = text if isinstance(text, list) else [text]
    tokens = [] if plain else ["[CLS]"]
    sents = []
    piece_to_word = []

    for sent in text:
        cnt = -1
        toks = _TOKENIZER.tokenize(sent)
        for tok in toks:
            if not tok.startswith('##'):
                cnt += 1
            piece_to_word.append(cnt)
        sents.append(toks)

    if max_tokens is not None:
        # Simple alternate truncation
        max_tokens = max_tokens - 1 - len(sents)
        while True:
            lengths = [len(toks) for toks in sents]
            total = sum(lengths)
            if total > max_tokens:
                sents[torch.argmax(torch.tensor(lengths)).item()].pop()
            else:
                break

    for toks in sents:
        tokens.extend(toks)
        if not plain:
            tokens.append("[SEP]")

    return tokens, piece_to_word


def tensorize(tokens, max_tokens=None, device=torch.device("cpu")):
    """Convert a list of tokens w/ padding.

    Args:
        tokens: a list of tokens
        max_tokens: max number of tokens per sentence
        device: target device storage
    """

    L = len(tokens)
    max_tokens = L if max_tokens is None else max_tokens
    tokens = stoi(tokens)
    if L <= max_tokens:
        padding = [PAD] * (max_tokens - L)
        tokens.extend(padding)
    else:
        logger.warning(f"Truncating length from {L} to {max_tokens}")
        padding = []
        tokens = tokens[:max_tokens]
        tokens[-1] = SEP
        L = max_tokens

    segment = []
    seg = 0
    prev = -1
    for i, tok in enumerate(tokens):
        if tok == SEP:
            segment.extend([seg] * (i - prev))
            seg += 1
            prev = i

    mask = [1] * L + padding
    segment.extend(padding)
    if not segment:
        segment.extend([0] * L)

    return (
        torch.tensor(tokens, device=device),
        torch.tensor(segment, device=device),
        torch.tensor(mask, device=device),
    )


def preprocess(text):
    # text.append('[SEP]')
    # print(f"preprocessed: {text}")
    return text


def postprocess(batch, vocab=None):
    """Convert a list of batch tokens into ids
    """
    # print(f"postprocess(): {batch}")
    ids = [stoi(ex) for ex in batch]
    return ids


"""
class BertVectors(Vectors):
    url = PRETRAINED_VOCAB_ARCHIVE_MAP
    
    def __init__(self, name='bert-base-uncased', **kwargs):
        url = self.url[name]
        super(BertVectors, self).__init__(name, url=url, **kwargs)

    def cache(self, name, cache, url=None, max_vectors=None):
        tokenizer = BertTokenizer.from_pretrained(name)
        self.stoi = tokenizer.vocab
        self.itos = tokenizer.ids_to_tokens
        
        model = BertModel.from_pretrained(name)
        self.vectors = model.embeddings.word_embedding.weight
        self.dim = model.config.hidden_size
        self.unk_init = None



class BertTextField(Field):
    def __init__(
        self,
        fix_length=64,
        lower=True,
        tokenizer_language="en",
        include_lengths=True,
        batch_first=True,
        pad_token="[PAD]",
        unk_token="[UNK]",
        pad_first=False,
        truncate_first=False,
        stop_words=None,
        is_target=False,
    ):
        super(BertTextField, self).__init__(
            sequential=True,
            use_vocab=False,
            init_token="[CLS]",
            # init_token=None,
            eos_token="[SEP]",
            fix_length=fix_length,
            preprocessing=preprocess,
            postprocessing=postprocess,
            lower=lower,
            tokenize=tokenize,
            # tokenizer_language=tokenizer_language,
            include_lengths=include_lengths,
            batch_first=batch_first,
            pad_token=pad_token,
            unk_token=unk_token,
            pad_first=pad_first,
            truncate_first=truncate_first,
            stop_words=stop_words,
            is_target=is_target,
        )
"""


def convert_snli(batch, device=torch.device("cpu")):
    segments = []
    masks = []
    for i in range(len(batch.premise[0])):
        premise, L0 = batch.premise[0][i], batch.premise[1][i]
        hypothesis, L1 = batch.hypothesis[0][i], batch.hypothesis[1][i]
        L = (L0 + L1 - 1).item()
        padding = [0] * (len(premise) - L)
        segment = [0] * L0.item() + [1] * (L1 - 1).item() + padding
        mask = [1] * L + padding

        premise[L0:L] = hypothesis[1:L1]
        segments.append(segment)
        masks.append(mask)
        # print(len(premise), L, len(segment), len(mask))

    # premise = batch.premise[0].to(device)
    # masks = th.tensor(masks, device=device)
    # segments = th.tensor(segments, device=device)
    # labels = batch.label.to(device)
    # return premise, masks, segments, labels
    return (
        batch.premise[0].to(device),
        torch.tensor(masks, device=device),
        torch.tensor(segments, device=device),
        batch.label.to(device),
    )


def save_vocabulary(path):
    """Save the tokenizer vocabulary to a directory or file."""
    index = 0
    vocab_file = os.path.join(path, VOCAB_NAME)
    with open(vocab_file, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(
                _TOKENIZER.vocab.items(), key=lambda kv: kv[1]
        ):
            if index != token_index:
                logger.warning(
                    "Saving vocabulary to {}: vocabulary indices are not consecutive."
                    " Please check that the vocabulary is not corrupted!".format(
                        vocab_file
                    )
                )
                index = token_index
            writer.write(token + "\n")
            index += 1
    return vocab_file


def save(model, path):
    output_dir = path if isinstance(path, Path) else Path(path)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), output_dir / WEIGHTS_NAME)
    # model_to_save.config.to_json_file(output_dir / CONFIG_NAME)
    with open(output_dir / CONFIG_NAME, "w", encoding="utf-8") as writer:
        writer.write(model_to_save.config.to_json_string())
    # _TOKENIZER.save_vocabulary(output_dir)
    save_vocabulary(output_dir)


def load(path, num_labels, cls="BertForSequenceClassification", lower=True):
    cls = getattr(BERT, cls)
    if cls is None:
        return None
    else:
        global _TOKENIZER
        model = cls.from_pretrained(path, num_labels=num_labels)
        _TOKENIZER = BertTokenizer.from_pretrained(path, do_lower_case=lower)
        return model
