#!/usr/local/bin/python3
import os
import json
import errno
import shutil
import re
import argparse


CONFIG_FILE_PATH = "config.json"


def get_config(cfg_file = CONFIG_FILE_PATH):
    with open(cfg_file) as f:
        return json.load(f)


def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)
    return


def parser():
    parser = argparse.ArgumentParser(description="File Utilities")
    parser.add_argument("--create", dest="create", action="store_true", default=None)
    parser.add_argument("--delete", dest="delete", action="store_true", default=None)
    return parser


def create_files(num_files):
    for i in range(num_files):
        from_directory = 'template_dir'
        to_directory = str(i)
        copy(from_directory, to_directory)


def delete_files():
    files = os.listdir(".")
    for f in files:
        if os.path.isdir(f) and re.match(r"^\d+$", f):
            shutil.rmtree(f)


if __name__ == '__main__':
    config = get_config()
    args = parser().parse_args()
    if args.create:
        create_files(config["num_birds"])
    elif args.delete:
        delete_files()
