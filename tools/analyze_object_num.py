import json
from collections import defaultdict

# COCO标注文件路径
ann_file = "/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/val4834-cocolike-cleaned-cat10.json"

with open(ann_file, "r", encoding="utf-8") as f:
    coco = json.load(f)

# category_id -> category_name
cat_id_to_name = {}

for cat in coco["categories"]:
    cat_id_to_name[cat["id"]] = cat["name"]

# 统计每个类别的实例数量
class_counts = defaultdict(int)

for ann in coco["annotations"]:
    cat_id = ann["category_id"]
    class_counts[cat_id] += 1

# 输出
print("=" * 50)
print("每个类别的目标数量")
print("=" * 50)

for cat_id, count in sorted(class_counts.items()):
    print(f"{cat_id:3d} | {cat_id_to_name[cat_id]:20s} | {count}")

'''
==================================================
train30000每个类别的目标数量
==================================================
  0 | normal               | 38103
  1 | ascus                | 26593
  2 | asch                 | 14678
  3 | lsil                 | 8899
  4 | hsil_scc_omn         | 11341
  5 | agc_adenocarcinoma_em | 11568
  6 | vaginalis            | 10995
  7 | monilia              | 2782
  8 | dysbacteriosis_herpes_act | 10968
  9 | ec                   | 10111   

==================================================
val4834每个类别的目标数量
==================================================
  0 | normal               | 6034
  1 | ascus                | 4364
  2 | asch                 | 2375
  3 | lsil                 | 1515
  4 | hsil_scc_omn         | 1832
  5 | agc_adenocarcinoma_em | 2070
  6 | vaginalis            | 1660
  7 | monilia              | 441
  8 | dysbacteriosis_herpes_act | 1463
  9 | ec                   | 1673
'''