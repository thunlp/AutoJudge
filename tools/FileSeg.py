#====================================================================================
# This file is used to perform (Chinese) word segmentation on a text file.
#
# Usage: python3 FileSeg.py file.txt
#
#====================================================================================

import sys
import jieba

print("Start to segment %s." % sys.argv[1])

with open(sys.argv[1], "r")as f:
    content = f.read().strip()
with open('after_segmentation_' + sys.argv[1], "w")as f:
    content = list(jieba.cut(content, cut_all=False))
    content = list(filter(lambda x: len(x) > 0, content))
    f.write(" ".join(content))

print("Finished segmentation %s." % sys.argv[1])
