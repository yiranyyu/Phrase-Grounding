import json
from pathlib import Path
import pprint


class Config(dict):
    def __init__(self, args=None, **kwargs):
        super(Config, self).__init__()
        if args is None:
            args = {}
        elif hasattr(args, '__dict__'):
            args = vars(args)

        args.update(kwargs)
        self.update(args)

    def update(self, other=None, **kwargs):
        if other is not None:
            kwargs = {**other, **kwargs}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self[key] = Config(value)
            else:
                self[key] = value
        return self

    def load(self, path):
        path = Path(path)
        if path.suffix == '.json':
            self.update(json.load(open(path)))
        else:
            raise ValueError(f'Unsupported config file format: {path.suffix}')

        return self

    def dump(self, path):
        return json.dump(self, open(path, 'w'))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__})'

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __eq__(self, c):
        return type(c) == type(self) and self.__dict__ == c.__dict__ or False

    def __bool__(self):
        return len(self) > 0

    def __contains__(self, key):
        return key in self.__dict__

    def __getattr__(self, key):
        return self[key]

    def __getitem__(self, key):
        return self.__dict__.get(key) if key in self else None

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.__dict__)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def get(self, *args):
        return self.__dict__.get(*args)
