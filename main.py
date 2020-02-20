from pathlib import Path

from numpy import format_float_positional as strf

import util.runner as runner
from util import logging
from util.app import run
from util.argparse import ArgumentParser
from util.config import Config


def parse_args(argv=None):
    parser = ArgumentParser(description="Grounding over Flickr30K Entities", fromfile_prefix_chars="@")
    parser.add_argument(
        "cmd",
        default="test",
        choices=["train", "test"],
        help="Command action to take",
    )

    # Dataset
    parser.add_argument("--data", default="data", help="Path to dataset")
    parser.add_argument("--dataset", default="flickr30k_entities", help="Dataset to train and test")
    parser.add_argument("--split", default="train,val", help="One or more dataset splits to select")
    parser.add_argument("--index", default=0, type=int, help="Index to an example in the split")
    parser.add_argument("--imgid", default=None, type=int, help="Image id in Flickr30K")
    parser.add_argument("--tok", default="wordpiece",
                        choices=["split", "wordpiece"], help="Plain string split or wordpiece tokenization.", )

    # System parameters
    parser.add_argument("-j", "--workers", type=int, default=0, help="number of threads to allocate")

    # Grounding task specific
    parser.add_argument("--arch", default="bert", choices=["bert"])
    parser.add_argument("--max-tokens", type=int, default=88, help="max number of caption tokens to allocate", )
    parser.add_argument("--max-entities", type=int, default=16, help="max number of caption entities to allocate", )
    parser.add_argument("--max-rois", type=int, default=100, help="max number of RoIs to allocate")

    # Dual BERT modalities for Contextual Grounding
    parser.add_argument("--bert-img-hidden-size", type=int, default=2048, help="Image embedding dimension")
    parser.add_argument("--bert-img-intermediate-size", type=int, default=3072,
                        help="Final image encoding dense dimension")
    parser.add_argument("--bert-img-layers", type=int, default=3, help="Number of image transformer layers", )
    parser.add_argument("--bert-img-heads", type=int, default=2, help="Number of image attention heads")
    parser.add_argument("--bert-img-hidden-dp", type=float, default=0.5, help="BERT hidden layer dropout prob")
    parser.add_argument("--bert-img-attention-dp", type=float, default=0.5, help="BERT attention dropout")
    parser.add_argument("--bert-img-spatial", type=str, default=None, choices=[None, 'abs', 'rel'],
                        help="BERT spatial encoding for images")

    # Training and Testing
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to enable 16-bit float precision instead of 32-bit.", )
    parser.add_argument("--epochs", type=int, default=7, help="Number of training epochs")
    parser.add_argument("--bs", type=str, default="256", help="batch size for each split")
    parser.add_argument("--grad-acc-steps", type=int, default=2,
                        help="Number of steps to accumulate gradients before an update pass to save memory.", )
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--warmup", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup e.g., 0.1 = 10%% of "
                             "training.", )
    parser.add_argument('--model_name_prefix', required=True, help='Name prefix of model to be stored')

    # Optimizer specific
    parser.add_argument("--optim", type=str, default="adam",
                        choices=["adam", "adamax"], help="Optimizer to train", )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--adam-wd", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam-eps", type=float, default=1e-6, help="Adam epsilon")
    parser.add_argument("--max-grad-norm", type=float, default=0.25, help="Max grad norm to clip")

    # Misc
    parser.add_argument("--log-interval", type=int, default=8,
                        help="frequency in batch iterations to show training progress", )
    parser.add_argument("--nsaved", type=int, default=3, help="Number of top performance models to save")
    parser.add_argument("--resume", type=str, default=None, help="path of saved model to resume")

    if argv is None:
        args = parser.parse_args()
    else:
        argv = type(argv) is str and argv.split() or argv
        args = parser.parse_args(argv)
    return Config(args=args)


def setup(config):
    """
        Essential setup for training:
        root: project directory
        data: path to dataset
        shared: path to NFS shared data
        model: formatted string combining architecture and configurations
        export: path to generate output
        save: path to save trained model, results, and logs
        split: one or more dataset splits to use
        bs: batch size for each split

    :param config: configuration to set up
    :return: configuration
    """

    config.root = Path(__file__).resolve().parent
    config.data = config.root / config.data
    config.model = f"{config.model_name_prefix}-s{config.seed}-{config.arch}-{config.optim}-" \
                   f"L{config.bert_img_layers}-H{config.bert_img_heads}-dp{config.bert_img_hidden_dp}-" \
                   f"b{config.bs}-lr{strf(config.lr)}-wp{config.warmup}" \
                   f"{'' if config.bert_img_spatial is None else f'-{config.bert_img_spatial}'}"
    config.export = config.root / "export"
    config.save = config.export / config.model
    config.save.mkdir(parents=True, exist_ok=True)

    if config.logfile is None:
        config.logfile = config.save / f'{config.model_name_prefix}.log'

    SPLITS = ['train', 'val', 'test']
    config.split = config.split.split(',')
    assert all(s in SPLITS for s in config.split)

    config.bs = list(map(int, config.bs.split(',')))
    if config.cmd == 'train':
        if len(config.split) == 1:
            config.split *= 2
        if len(config.bs) == 1:
            config.bs *= 2
    else:
        config.split = config.split[0]
        config.bs = config.bs[0]

    return config


def main(config):
    runner.run(config)


if __name__ == "__main__":
    raw_config = parse_args()
    run(main, setup(raw_config))
