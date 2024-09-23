# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse

from modelscope.cli.download import DownloadCMD
from modelscope.cli.modelcard import ModelCardCMD
from modelscope.cli.pipeline import PipelineCMD
from modelscope.cli.plugins import PluginsCMD


def run_cmd():
    parser = argparse.ArgumentParser(
        'ModelScope Command Line tool', usage='modelscope <command> [<args>]')
    subparsers = parser.add_subparsers(help='modelscope commands helpers')

    DownloadCMD.define_args(subparsers)
    PluginsCMD.define_args(subparsers)
    PipelineCMD.define_args(subparsers)
    ModelCardCMD.define_args(subparsers)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    cmd = args.func(args)
    cmd.execute()


if __name__ == '__main__':
    run_cmd()
