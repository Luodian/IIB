import os
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(
    description="Domain generalization testbed")
parser.add_argument("--input_dir", type=str, required=True)
args = parser.parse_args()

path_list = os.listdir(args.input_dir)

for item in path_list:
    c_path = os.path.join(args.input_dir, item)
    print(c_path)
    print("*" * 60)
    cmd = "/home/v-boli4/miniconda3/envs/torch/bin/python -u collect_results.py --input_dir {}".format(c_path)
    subprocess.call(cmd, shell=True)
