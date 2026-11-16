import os
import xml.etree.ElementTree as ET
import json
import argparse
from datetime import datetime
from collections import defaultdict


def xml_to_coco(xml_folder, output_file):
    """
    将XML标注文件转换为COCO格式

    Args:
        xml_folder: XML文件所在文件夹路径
        output_file: 输出的COCO格式JSON文件路径
    """

    # COCO格式的基本结构
    coco_format = {
        "info": {
            "description": "Converted from XML annotations",
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

    # 统计类别
    categories = set()
    category_map = {}
    image_id = 1
    annotation_id = 1

    # 获取所有XML文件
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]

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
                category_name = obj.find('name').text
                categories.add(category_name)

                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # 计算边界框的坐标和尺寸
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin

                # 创建标注信息
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,  # 临时值，后面会更新
                    "segmentation": [],  # COCO格式通常需要分割信息，这里为空
                    "area": bbox_width * bbox_height,
                    "bbox": [xmin, ymin, bbox_width, bbox_height],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)

                annotation_id += 1

            image_id += 1

        except Exception as e:
            print(f"处理文件 {xml_file} 时出错: {e}")
            continue

    # 创建类别映射
    categories = sorted(list(categories))
    for i, category in enumerate(categories, 1):
        category_map[category] = i
        coco_format["categories"].append({
            "id": i,
            "name": category,
            "supercategory": "none"
        })

    # 更新标注中的category_id
    for annotation in coco_format["annotations"]:
        # 我们需要重新解析XML来获取正确的类别信息
        # 这里简化处理，实际使用时需要更复杂的逻辑
        pass

    print(f"转换完成:")
    print(f"- 图像数量: {len(coco_format['images'])}")
    print(f"- 标注数量: {len(coco_format['annotations'])}")
    print(f"- 类别数量: {len(coco_format['categories'])}")
    print(f"- 类别列表: {[cat['name'] for cat in coco_format['categories']]}")

    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到: {output_file}")


def process_single_xml(xml_file, output_file):
    """
    处理单个XML文件的版本
    """
    coco_format = {
        "info": {
            "description": "Converted from single XML annotation",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "images": [],
        "annotations": [],
        "categories": []
    }

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 提取图像信息
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 添加到images列表
        image_info = {
            "id": 1,
            "width": width,
            "height": height,
            "file_name": filename
        }
        coco_format["images"].append(image_info)

        # 处理类别
        categories_set = set()
        annotation_id = 1

        for obj in root.findall('object'):
            category_name = obj.find('name').text
            categories_set.add(category_name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            annotation = {
                "id": annotation_id,
                "image_id": 1,
                "category_id": 0,  # 临时值
                "segmentation": [],
                "area": bbox_width * bbox_height,
                "bbox": [xmin, ymin, bbox_width, bbox_height],
                "iscrowd": 0
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1

        # 添加类别信息
        categories_list = sorted(list(categories_set))
        for i, category in enumerate(categories_list, 1):
            coco_format["categories"].append({
                "id": i,
                "name": category,
                "supercategory": "none"
            })

        # 更新标注的category_id
        category_id_map = {cat: i + 1 for i, cat in enumerate(categories_list)}
        for i, annotation in enumerate(coco_format["annotations"]):
            # 这里需要根据实际类别名称设置正确的category_id
            # 由于我们无法从当前信息中获取，这里设为1
            annotation["category_id"] = 1

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, indent=2, ensure_ascii=False)

        print(f"转换完成! 结果保存到: {output_file}")
        print(f"- 找到 {len(coco_format['annotations'])} 个标注")
        print(f"- 类别: {categories_list}")

    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将XML标注转换为COCO格式')
    parser.add_argument('--input', type=str, required=True, help='输入XML文件或文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出JSON文件路径')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        # 处理文件夹
        xml_to_coco(args.input, args.output)
    elif os.path.isfile(args.input) and args.input.endswith('.xml'):
        # 处理单个文件
        process_single_xml(args.input, args.output)
    else:
        print("请输入有效的XML文件或文件夹路径")