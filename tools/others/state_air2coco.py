import os
import xml.etree.ElementTree as ET
import json
import argparse
from datetime import datetime
from collections import defaultdict


def correct_category_name(category_name):
    """
    修正类别名称中的错误，并过滤无效类别
    """
    # 修正拼写错误
    if category_name == "persaon":
        return "person"

    # 处理可能的数字后缀
    if category_name == "dog0":
        return "dog"

    # 有效的类别列表
    valid_categories = ["baby carriage", "bicycle", "boat", "bus", "car",
                        "dog", "motorcycle", "person", "sports ball", "truck"]

    # 如果是有效类别，直接返回；否则返回None表示过滤掉
    if category_name in valid_categories:
        return category_name
    else:
        print(f"警告: 发现无效类别 '{category_name}'，已过滤")
        return None


def xml_to_coco(xml_folder, output_file):
    """
    将XML标注文件转换为COCO格式，并修正类别名称

    Args:
        xml_folder: XML文件所在文件夹路径
        output_file: 输出的COCO格式JSON文件路径
    """

    # COCO格式的基本结构
    coco_format = {
        "info": {
            "description": "Converted from XML annotations with category correction",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "url": "",
                "id": 1,
                "name": "Unknown"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 预定义的10个类别
    predefined_categories = [
        {"id": 1, "name": "baby carriage", "supercategory": "vehicle"},
        {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
        {"id": 3, "name": "boat", "supercategory": "vehicle"},
        {"id": 4, "name": "bus", "supercategory": "vehicle"},
        {"id": 5, "name": "car", "supercategory": "vehicle"},
        {"id": 6, "name": "dog", "supercategory": "animal"},
        {"id": 7, "name": "motorcycle", "supercategory": "vehicle"},
        {"id": 8, "name": "person", "supercategory": "human"},
        {"id": 9, "name": "sports ball", "supercategory": "object"},
        {"id": 10, "name": "truck", "supercategory": "vehicle"}
    ]

    # 创建类别名称到ID的映射
    category_name_to_id = {cat["name"]: cat["id"] for cat in predefined_categories}

    # 添加类别到COCO格式
    coco_format["categories"] = predefined_categories

    # 统计信息
    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "filtered_annotations": 0,
        "category_count": defaultdict(int),
        "corrected_categories": 0
    }

    image_id = 1
    annotation_id = 1

    # 获取所有XML文件
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    xml_files.sort()  # 按文件名排序以保证一致性

    print(f"找到 {len(xml_files)} 个XML文件")

    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)

        try:
            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 提取图像信息
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # 添加到images列表
            image_info = {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": filename,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            }
            coco_format["images"].append(image_info)

            # 处理每个标注对象
            for obj in root.findall('object'):
                original_name = obj.find('name').text
                corrected_name = correct_category_name(original_name)

                # 如果类别无效，跳过此标注
                if corrected_name is None:
                    stats["filtered_annotations"] += 1
                    continue

                # 统计修正情况
                if original_name != corrected_name:
                    stats["corrected_categories"] += 1
                    print(f"修正: {original_name} -> {corrected_name} (文件: {xml_file})")

                # 获取边界框信息
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # 计算边界框的坐标和尺寸
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin

                # 确保边界框在图像范围内
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)
                bbox_width = max(0, xmax - xmin)
                bbox_height = max(0, ymax - ymin)

                # 如果边界框无效，跳过此标注
                if bbox_width <= 0 or bbox_height <= 0:
                    stats["filtered_annotations"] += 1
                    print(f"警告: 跳过无效边界框 (文件: {xml_file})")
                    continue

                # 创建标注信息
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_name_to_id[corrected_name],
                    "segmentation": [],  # COCO格式通常需要分割信息，这里为空
                    "area": bbox_width * bbox_height,
                    "bbox": [xmin, ymin, bbox_width, bbox_height],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)

                # 更新统计信息
                stats["total_annotations"] += 1
                stats["category_count"][corrected_name] += 1
                annotation_id += 1

            stats["total_images"] += 1
            image_id += 1

        except Exception as e:
            print(f"处理文件 {xml_file} 时出错: {e}")
            continue

    # 打印统计信息
    print("\n转换统计信息:")
    print(f"- 成功处理的图像数量: {stats['total_images']}")
    print(f"- 总标注数量: {stats['total_annotations']}")
    print(f"- 过滤的无效标注数量: {stats['filtered_annotations']}")
    print(f"- 修正的类别数量: {stats['corrected_categories']}")
    print("- 各类别标注数量:")
    for category in predefined_categories:
        category_name = category["name"]
        count = stats["category_count"][category_name]
        print(f"  - {category_name}: {count}")

    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2, ensure_ascii=False)

    print(f"\nCOCO格式标注已保存到: {output_file}")
    return coco_format


def validate_coco_json(json_file):
    """
    验证生成的COCO JSON文件是否符合格式要求
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            if key not in coco_data:
                print(f"错误: 缺少必需的键 '{key}'")
                return False

        # 验证类别数量
        expected_category_count = 10
        if len(coco_data["categories"]) != expected_category_count:
            print(f"错误: 类别数量应为{expected_category_count}个，实际为{len(coco_data['categories'])}个")
            return False

        # 验证类别名称
        expected_categories = {"baby carriage", "bicycle", "boat", "bus", "car",
                               "dog", "motorcycle", "person", "sports ball", "truck"}
        actual_categories = {cat["name"] for cat in coco_data["categories"]}

        if expected_categories != actual_categories:
            print(f"错误: 类别名称不匹配")
            print(f"期望: {expected_categories}")
            print(f"实际: {actual_categories}")
            return False

        print("COCO JSON文件验证通过!")
        print(f"- 图像数量: {len(coco_data['images'])}")
        print(f"- 标注数量: {len(coco_data['annotations'])}")
        print(f"- 类别数量: {len(coco_data['categories'])}")
        print(f"- 类别列表: {[cat['name'] for cat in coco_data['categories']]}")

        return True

    except Exception as e:
        print(f"验证文件时出错: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将XML标注转换为COCO格式（自动修正类别名称，仅包含10个有效类别）')
    parser.add_argument('--input', type=str, required=True, help='输入XML文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出JSON文件路径')
    parser.add_argument('--validate', action='store_true', help='转换后验证输出文件')

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print("错误: 输入路径必须是一个包含XML文件的文件夹")
        exit(1)

    result = xml_to_coco(args.input, args.output)

    # 验证输出文件
    if args.validate and os.path.exists(args.output):
        validate_coco_json(args.output)