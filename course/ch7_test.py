#!/usr/bin/env python3
"""Ch7 测试运行：3类版本"""
import sys
sys.path.insert(0, "/root/gpufree-data")
# 读取原脚本，替换类别列表为3类测试
with open("/root/gpufree-data/ch7_pipeline.py") as f:
    code = f.read()
code = code.replace(
    'all_categories = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())',
    'all_categories = ["D_sub_connector", "3_adapter", "DVD_switch"]  # TEST'
)
exec(code)
