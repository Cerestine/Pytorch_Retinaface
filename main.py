"""Main execution file"""
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--mode', default="train", type=str, help='Set app flag(train/detect/convert), Default is train')
parser.add_argument('--gpu_num', default=1, type=int, help='Set gpu flag, Default is 1, 0 for CPU')
parser.add_argument('--config_file', default="config.json", type=str, help='Config file, Default is config.json')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == "train":
        from app.train import run_train as app

    app(args.config_file)
    