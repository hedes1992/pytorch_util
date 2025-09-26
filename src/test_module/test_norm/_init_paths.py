#coding=utf-8
import os.path as osp
import sys

root_dir = osp.abspath(__file__).split("test_module")[0]
sys.path.insert(0, root_dir)
print(f"sys.path: {sys.path}")