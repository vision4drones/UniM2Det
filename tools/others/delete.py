import os
import json
import argparse
from collections import defaultdict


def clean_coco_annotations(images_folder, json_file, output_file=None):
    """
    清理COCO标注文件，删除不存在的图片相关信息

    Args:
        images_folder: 图片文件夹路径
        json_file: COCO格式的JSON标注文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """

    # 如果没有指定输出文件，则覆盖原文件
    if output_file is None:
        output_file = json_file

    # 读取COCO标注文件
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return False

    # 检查必需的键
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in coco_data:
            print(f"错误: JSON文件中缺少必需的键 '{key}'")
            return False

    print("开始检查图片文件是否存在...")

    # 获取图片文件夹中的所有图片文件
    image_files = set()
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        for file in os.listdir(images_folder):
            if file.lower().endswith(ext):
                image_files.add(file)

    print(f"图片文件夹中找到 {len(image_files)} 个图片文件")

    # 检查每张图片是否存在
    valid_images = []
    missing_images = []

    for img_info in coco_data["images"]:
        filename = img_info["file_name"]

        # 检查文件是否存在
        if filename in image_files:
            valid_images.append(img_info)
        else:
            missing_images.append(filename)

    print(f"\n检查结果:")
    print(f"- 有效的图片: {len(valid_images)}")
    print(f"- 缺失的图片: {len(missing_images)}")

    if missing_images:
        print("\n缺失的图片文件:")
        for i, filename in enumerate(missing_images[:10]):  # 只显示前10个
            print(f"  {i + 1}. {filename}")
        if len(missing_images) > 10:
            print(f"  ... 还有 {len(missing_images) - 10} 个缺失文件")

    # 构建有效的图片ID集合
    valid_image_ids = {img["id"] for img in valid_images}

    # 过滤标注，只保留有效图片的标注
    valid_annotations = []
    for ann in coco_data["annotations"]:
        if ann["image_id"] in valid_image_ids:
            valid_annotations.append(ann)

    print(f"\n标注过滤结果:")
    print(f"- 原始标注数量: {len(coco_data['annotations'])}")
    print(f"- 有效标注数量: {len(valid_annotations)}")
    print(f"- 删除的标注数量: {len(coco_data['annotations']) - len(valid_annotations)}")

    # 更新COCO数据
    coco_data["images"] = valid_images
    coco_data["annotations"] = valid_annotations

    # 保存清理后的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"\n清理完成! 结果已保存到: {output_file}")
        return True

    except Exception as e:
        print(f"保存文件失败: {e}")
        return False


def analyze_coco_json(json_file):
    """
    分析COCO JSON文件的结构
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        print("COCO文件分析结果:")
        print(f"- 图片数量: {len(coco_data.get('images', []))}")
        print(f"- 标注数量: {len(coco_data.get('annotations', []))}")
        print(f"- 类别数量: {len(coco_data.get('categories', []))}")

        # 统计每个类别的标注数量
        if "annotations" in coco_data and "categories" in coco_data:
            category_counts = defaultdict(int)
            for ann in coco_data["annotations"]:
                category_counts[ann["category_id"]] += 1

            print("- 各类别标注数量:")
            category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
            for cat_id, count in category_counts.items():
                cat_name = category_id_to_name.get(cat_id, f"未知({cat_id})")
                print(f"  - {cat_name}: {count}")

    except Exception as e:
        print(f"分析文件失败: {e}")


def find_missing_images_only(images_folder, json_file):
    """
    只查找缺失的图片，不进行删除操作
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # 获取图片文件夹中的所有图片文件
        image_files = set()
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            for file in os.listdir(images_folder):
                if file.lower().endswith(ext):
                    image_files.add(file)

        missing_images = []
        for img_info in coco_data["images"]:
            filename = img_info["file_name"]
            if filename not in image_files:
                missing_images.append({
                    "filename": filename,
                    "image_id": img_info["id"]
                })

        return missing_images

    except Exception as e:
        print(f"查找缺失图片失败: {e}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='清理COCO标注文件中不存在的图片信息')
    parser.add_argument('--images', type=str, required=True, help='图片文件夹路径')
    parser.add_argument('--json', type=str, required=True, help='COCO格式的JSON标注文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径，如果不指定则覆盖原文件')
    parser.add_argument('--analyze', action='store_true', help='分析JSON文件结构')
    parser.add_argument('--find-only', action='store_true', help='只查找缺失图片，不进行删除')

    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.images):
        print(f"错误: 图片文件夹不存在: {args.images}")
        exit(1)

    if not os.path.exists(args.json):
        print(f"错误: JSON文件不存在: {args.json}")
        exit(1)
 
    if args.analyze:
        print("=== 分析JSON文件 ===")
        analyze_coco_json(args.json)

    elif args.find_only:
        print("=== 查找缺失图片 ===")
        missing_images = find_missing_images_only(args.images, args.json)
        if missing_images:
            print(f"找到 {len(missing_images)} 个缺失的图片:")
            for img in missing_images:
                print(f"  - 图片ID: {img['image_id']}, 文件名: {img['filename']}")
        else:
            print("没有找到缺失的图片")

    else:
        print("=== 开始清理COCO标注文件 ===")
        success = clean_coco_annotations(args.images, args.json, args.output)

        if success and args.output:
            print(f"\n=== 清理后的文件分析 ===")
            analyze_coco_json(args.output)