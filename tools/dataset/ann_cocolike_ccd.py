import json
import os

def update_coco_file_names(input_json_path, output_json_path):
    """
    读取COCO格式的JSON文件，更新其中的file_name字段为只包含文件名，
    并保存修改后的JSON文件。
    
    :param input_json_path: 输入的原始COCO标注文件路径
    :param output_json_path: 输出的修改后的COCO标注文件路径
    """
    # 读取原始的COCO标注文件
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # 遍历所有的图像信息并更新file_name
    for img_info in coco_data['images']:
        full_file_name = img_info['file_name']
        file_name = os.path.basename(full_file_name)  # 获取文件名
        img_info['file_name'] = file_name  # 更新file_name

    # 保存修改后的COCO标注文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

# input_json_path = '/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/train30000-cat10.json'
# output_json_path = '/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/train30000-cocolike-cat10.json'

# update_coco_file_names(input_json_path, output_json_path)

# input_json_path = '/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/val5000-cat10.json'
# output_json_path = '/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/val5000-cocolike-cat10.json'

input_json_path = '/root/commonfile/TCT_JPEGImages/test10000-cat10.json'
output_json_path = '/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/test10000-cocolike-cat10.json'

update_coco_file_names(input_json_path, output_json_path)