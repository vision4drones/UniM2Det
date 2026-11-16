import json
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def parse_single_txt(txt_file_path):
    """
    解析单个txt标注文件
    返回: list of dicts, 每个dict包含bbox和category
    """
    annotations = []
    with open(txt_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            # 提取8个坐标点
            coords = list(map(float, parts[:8]))
            category = parts[8]
            difficult = int(parts[9])

            # 从8个点中提取HBB (x_min, y_min, x_max, y_max)
            x_coords = coords[0::2]  # 提取所有x坐标
            y_coords = coords[1::2]  # 提取所有y坐标

            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            annotations.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'category': category,
                'difficult': difficult
            })

    return annotations


def convert_to_coco(txt_dir, image_dir, output_json_path, category_mapping=None):
    """
    将多个txt标注文件转换为单个COCO格式JSON文件

    参数:
        txt_dir: 包含txt标注文件的目录
        image_dir: 包含图像文件的目录
        output_json_path: 输出的COCO JSON文件路径
        category_mapping: 类别映射字典，如果为None则自动生成
    """

    # 获取所有txt文件
    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files:
        print(f"错误: 在 {txt_dir} 中没有找到txt文件")
        return None

    print(f"找到 {len(txt_files)} 个标注文件")

    # 第一遍：收集所有类别
    all_categories = set()
    file_annotations = {}  # 存储每个文件的标注

    for txt_file in tqdm(txt_files, desc="读取标注文件"):
        annotations = parse_single_txt(txt_file)
        file_annotations[txt_file.stem] = annotations
        for ann in annotations:
            all_categories.add(ann['category'])

    # 创建类别映射
    if category_mapping is None:
        category_mapping = {cat: idx + 1 for idx, cat in enumerate(sorted(all_categories))}

    print(f"类别映射: {category_mapping}")

    # 准备COCO数据结构
    coco_data = {
        "info": {
            "description": "Converted from custom HBB format to COCO",
            "version": "1.0",
            "year": 2024,
            "contributor": "Conversion Script",
            "date_created": "2024/01/01"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cat_id, "name": cat_name, "supercategory": "object"}
            for cat_name, cat_id in category_mapping.items()
        ]
    }

    # 处理每张图片
    image_id = 1
    annotation_id = 1

    for img_name, annotations in tqdm(file_annotations.items(), desc="转换标注"):
        # 查找图像文件（支持常见格式）
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff' ,'.npy']:
            candidate = Path(image_dir) / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"警告: 找不到图像 {img_name}，跳过...")
            continue

        # 获取图像尺寸
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"错误: 无法读取图像 {img_path}: {e}")
            continue

        # 添加图像信息
        coco_data["images"].append({
            "id": image_id,
            "width": img_width,
            "height": img_height,
            "file_name": img_path.name,
            "license": None,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": ""
        })

        # 添加该图像的所有标注
        for ann in annotations:
            x_min, y_min, x_max, y_max = ann['bbox']

            # 计算COCO格式的bbox: [x, y, width, height]
            width = x_max - x_min
            height = y_max - y_min

            # 验证bbox有效性
            if width <= 0 or height <= 0:
                print(f"警告: 图像 {img_name} 中有无效bbox: {ann['bbox']}，跳过...")
                continue

            # 确保bbox在图像范围内
            x_min = max(0, min(x_min, img_width))
            y_min = max(0, min(y_min, img_height))
            width = min(width, img_width - x_min)
            height = min(height, img_height - y_min)

            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_mapping[ann['category']],
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "segmentation": [],  # 空列表表示使用bbox作为分割
                "iscrowd": 0,
                "attributes": {
                    "difficult": ann['difficult']  # 保留难易度信息
                }
            }

            coco_data["annotations"].append(coco_annotation)
            annotation_id += 1

        image_id += 1

    # 保存COCO JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # 打印统计信息
    print("\n" + "=" * 50)
    print("转换完成！")
    print(f"输出文件: {output_json_path}")
    print(f"图像数量: {len(coco_data['images'])}")
    print(f"标注数量: {len(coco_data['annotations'])}")
    print(f"类别数量: {len(coco_data['categories'])}")
    print(f"类别详情: {category_mapping}")

    # 按类别统计标注数量
    category_stats = {}
    for ann in coco_data['annotations']:
        cat_name = [c['name'] for c in coco_data['categories'] if c['id'] == ann['category_id']][0]
        category_stats[cat_name] = category_stats.get(cat_name, 0) + 1

    print("\n各类别标注统计:")
    for cat_name, count in category_stats.items():
        print(f"  {cat_name}: {count} 个标注")
    print("=" * 50)

    return coco_data


def split_coco_dataset(coco_json_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    将COCO数据集分割为训练集、验证集和测试集
    """
    import random
    from sklearn.model_selection import train_test_split

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 获取所有图像ID
    image_ids = [img['id'] for img in coco_data['images']]

    # 分割数据集
    train_ids, temp_ids = train_test_split(image_ids, test_size=(1 - train_ratio), random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

    # 创建分割后的数据集
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    # 保存分割后的数据集
    base_path = Path(coco_json_path).parent
    for split_name, split_ids in splits.items():
        split_data = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "categories": coco_data["categories"],
            "images": [img for img in coco_data['images'] if img['id'] in split_ids],
            "annotations": [ann for ann in coco_data['annotations'] if ann['image_id'] in split_ids]
        }

        split_path = base_path / f"{Path(coco_json_path).stem}_{split_name}.json"
        with open(split_path, 'w') as f:
            json.dump(split_data, f, indent=2)

        print(f"{split_name}集: {len(split_data['images'])} 张图片, {len(split_data['annotations'])} 个标注")
        print(f"保存至: {split_path}")

    return splits


# 使用示例
if __name__ == "__main__":
    # 设置路径
    txt_directory = "/home/ldd/newdisk/MODA/test/labels/"  # 存放txt标注文件的目录
    image_directory = "/home/ldd/newdisk/MODA/test/images/"  # 存放图像文件的目录
    output_json = "/home/ldd/newdisk/MODA/coco_anno/test.json"  # 输出的COCO JSON文件

    # 方式1：自动检测类别
    convert_to_coco(
        txt_dir=txt_directory,
        image_dir=image_directory,
        output_json_path=output_json
    )

    # 方式2：指定类别映射（可选）
    # custom_category_mapping = {
    #     "car": 1,
    #     "truck": 2,
    #     "bike": 3,
    #     "pedestrian": 4
    # }
    #
    # convert_to_coco(
    #     txt_dir=txt_directory,
    #     image_dir=image_directory,
    #     output_json_path=output_json,
    #     category_mapping=custom_category_mapping
    # )

    # 方式3：分割数据集（可选）
    # split_coco_dataset(output_json, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)