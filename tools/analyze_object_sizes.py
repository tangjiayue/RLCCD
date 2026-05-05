import json

def analyze_resized_sizes(json_path, input_size=640, original_size=(4112, 3008)):
    """
    分析目标在 Resize 到 input_size (640x640) 后的大小分布。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. 计算缩放比例 (Scale Factor)
    # 假设保持长宽比缩放，以短边为准或者直接resize。
    # 这里为了简化估算，我们计算面积缩放比例。
    # 原始面积 vs 目标输入面积
    orig_w, orig_h = original_size
    orig_area_total = orig_w * orig_h
    input_area_total = input_size * input_size
    
    # 面积缩放因子 (例如：640x640 / 4112x3008)
    scale_factor = input_area_total / orig_area_total
    
    print(f"原始分辨率: {original_size}")
    print(f"输入分辨率: {input_size}x{input_size}")
    print(f"面积缩放比例: {scale_factor:.4f}")
    print("-" * 30)

    small = 0
    medium = 0
    large = 0
    
    # COCO 标准定义 (基于 640x640 输入)
    # 小目标: area < 32^2 (1024)
    # 中目标: 32^2 <= area < 96^2 (9216)
    # 大目标: area >= 96^2
    
    for ann in data['annotations']:
        # 2. 将原始面积映射到 640x640 尺度
        orig_area = ann['area']
        resized_area = orig_area * scale_factor
        
        if resized_area < 1024:
            small += 1
        elif resized_area < 9216:
            medium += 1
        else:
            large += 1
    
    total = small + medium + large
    
    print("=" * 50)
    print("目标大小分布统计 (映射到 640x640)")
    print("=" * 50)
    print(f"小目标 (面积 < 1024): {small} 个 ({small/total*100:.1f}%)")
    print(f"中目标 (1024 ≤ 面积 < 9216): {medium} 个 ({medium/total*100:.1f}%)")
    print(f"大目标 (面积 ≥ 9216): {large} 个 ({large/total*100:.1f}%)")
    print(f"总计: {total} 个目标")
    print("=" * 50)
    
    return small, medium, large

if __name__ == "__main__":
    # json_file = "/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/train30000-cocolike-cat10.json"
    json_file = "/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/val5000-cocolike-cat10.json"
    # 注意：这里传入了原始尺寸 (4112, 3008)
    analyze_resized_sizes(json_file, input_size=640, original_size=(4112, 3008))

'''
(rlccd) root@6efbe0cde6bd ~/u/P/RLCCD# python /root/userfolder/Projects/RLCCD/tools/analyze_object_sizes.py                      原始分辨率: (4112, 3008)
输入分辨率: 640x640
面积缩放比例: 0.0331
------------------------------
==================================================
目标大小分布统计 (映射到 640x640)
==================================================
小目标 (面积 < 1024): 58899 个 (40.3%)
中目标 (1024 ≤ 面积 < 9216): 83515 个 (57.2%)
大目标 (面积 ≥ 9216): 3624 个 (2.5%)
总计: 146038 个目标
==================================================
(rlccd) root@6efbe0cde6bd ~/u/P/RLCCD# python /root/userfolder/Projects/RLCCD/tools/analyze_object_sizes.py
原始分辨率: (4112, 3008)
输入分辨率: 640x640
面积缩放比例: 0.0331
------------------------------
==================================================
目标大小分布统计 (映射到 640x640)
==================================================
小目标 (面积 < 1024): 9996 个 (41.1%)
中目标 (1024 ≤ 面积 < 9216): 13719 个 (56.5%)
大目标 (面积 ≥ 9216): 586 个 (2.4%)
总计: 24301 个目标
==================================================
'''
