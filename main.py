"""Main execution file"""
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--mode', default="detect", type=str, help='Set app flag(train/detect/convert), Default is train')
parser.add_argument('--gpu_num', default=1, type=int, help='Set gpu flag, Default is 1, 0 for CPU')
parser.add_argument('--config_file', default="detect_config.json", type=str, help='Config file, Default is train_config.json')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == "convert":
        from app.onnx_converter import onnx_convertion as app
    elif args.mode == "detect":
        from app.detect import detect as app
    else:
        from app.train import train as app
    app(args.config_file)
    