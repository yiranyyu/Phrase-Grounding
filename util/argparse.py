import argparse
import re

from util.config import Config


class ConfigAction(argparse.Action):
    def __call__(self, parser, args, path, **kwargs):
        """
        Parse and combine command line options with a configuration file if specified.
        :param **kwargs:
        :param parser:
        :param args:
        :param path:
        """
        cfg = Config()
        cfg.load(path)
        for k, v in vars(cfg).items():
            field = args.__dict__.get(k)
            if isinstance(field, Config):
                field.update(v)
            else:
                args.__dict__[k] = v


class ArgumentParser(argparse.ArgumentParser):
    r"""Allow no '=' in an argument config as a regular command line
    """

    def __init__(self, *args, **kwargs):
        char = kwargs.get("fromfile_prefix_chars", '@')
        kwargs["fromfile_prefix_chars"] = char
        super(ArgumentParser, self).__init__(*args, **kwargs)
        self.add_argument("--cfg", action=ConfigAction, help="path to a configuration file in JSON")

        # App/Logging
        self.add_argument('--logfile', default=None, help='Path to log messages if provided')
        self.add_argument('--log_std', action='store_true', help='Print log to stdout')

        # CUDA
        self.add_argument('-s', '--seed', default='1204', type=int, help='Random seed')
        self.add_argument('--deterministic', action='store_true', help='Deterministic training results')
        self.add_argument('--gpu', nargs='+', type=int,
                          help='One or more visible GPU indices as CUDA_VISIBLE_DEVICES w/o comma')
        self.add_argument('--no-gpu', action='store_true', help='Only use CPU regardless of --gpu')

    def convert_arg_line_to_args(self, line):
        tokens = line.strip().split()
        first = re.split(r"=", tokens[0])
        return first + tokens[1:]

    def parse_args(self, args=None, ns=None):
        args = super(ArgumentParser, self).parse_args(args, ns)
        cfg = Config(args)
        return cfg

    def parse_known_args(self, args=None, ns=None):
        args, leftover = super(ArgumentParser, self).parse_known_args(args, ns)
        cfg = Config(args)
        return cfg, leftover
