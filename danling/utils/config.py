from argparse import Namespace
from ast import literal_eval


class Config(Namespace):
    def __setattr__(self, name, value):
        try:
            value = literal_eval(value)
        except ValueError:
            pass
        if '.' in name:
            name = name.split('.')
            name, rest = name[0], '.'.join(name[1:])
            setattr(self, name, type(self)())
            setattr(getattr(self, name), rest, value)
        else:
            super().__setattr__(name, value)

    def dict(self):
        dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                dict[k] = v.dict()
            else:
                dict[k] = v
        return dict
