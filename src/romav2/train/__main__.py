"""Command line entry point: ``python -m romav2.train --stage matcher --megadepth data/megadepth``.

Every field of ``TrainCfg`` is exposed as ``--<field>`` (underscores become dashes).
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING, fields

from romav2.train.train import TrainCfg, train


def parse_args() -> TrainCfg:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(TrainCfg):
        name = "--" + f.name.replace("_", "-")
        default = f.default if f.default is not MISSING else f.default_factory()
        if f.type == "bool":
            parser.add_argument(name, action="store_true", default=default)
        elif f.type == "list[str]":
            parser.add_argument(name, nargs="*", default=default)
        else:
            base = str if f.default is None else type(default)
            parser.add_argument(name, type=base, default=default)
    return TrainCfg(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
