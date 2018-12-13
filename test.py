from collections import namedtuple
import argparse


class ProgramArgs(argparse.Namespace):
    def __init__(self):
        super(ProgramArgs, self).__init__()
        self.num_layer = 20


parser = argparse.ArgumentParser()
nsp = ProgramArgs()
for key, value in nsp.__dict__.items():
    parser.add_argument('-{}'.format(key),
                        action='store',
                        default=value,
                        type=type(value),
                        dest=str(key))
config = parser.parse_args(namespace=nsp)  # type: ProgramArgs

print(config.num_layer)
