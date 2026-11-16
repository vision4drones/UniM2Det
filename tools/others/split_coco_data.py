import json
import os
import random
import shutil
from collections import defaultdict
import argparse


def split_coco_dataset(coco_annotation_file, output_dir, train_ratio=0.8, seed=42):
    """
    将COCO格式数据集划分为训练集和测试集

    Args:
        coco_annotation_file: COCO标注文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子
    """

    # 设置随机种子
    random.seed(seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    # 读取COCO标注文件
    with open(coco_annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    print("原始数据集统计:")
    print(f"图像数量: {len(coco_data['images'])}")
    print(f"标注数量: {len(coco_data['annotations'])}")
    print(f"类别数量: {len(coco_data['categories'])}")

    # 获取所有图像ID
    image_ids = [img['id'] for img in coco_data['images']]
    random.shuffle(image_ids)

    # 划分训练集和测试集
    split_index = int(len(image_ids) * train_ratio)
    train_image_ids = set(image_ids[:split_index])
    val_image_ids = set(image_ids[split_index:])

    print(f"\n划分结果:")
    print(f"训练集图像数量: {len(train_image_ids)}")
    print(f"测试集图像数量: {len(val_image_ids)}")

    # 构建训练集标注
    train_annotations = {
        "images": [],
        "annotations": [],
        "categories": coco_data['categories']
    }

    # 构建测试集标注
    val_annotations = {
        "images": [],
        "annotations": [],
        "categories": coco_data['categories']
    }

    # 处理图像信息
    for image in coco_data['images']:
        if image['id'] in train_image_ids:
            train_annotations['images'].append(image)
        else:
            val_annotations['images'].append(image)

    # 处理标注信息
    for annotation in coco_data['annotations']:
        if annotation['image_id'] in train_image_ids:
            train_annotations['annotations'].append(annotation)
        else:
            val_annotations['annotations'].append(annotation)

    # 保存划分后的标注文件
    train_output = os.path.join(output_dir, 'annotations', 'instances_train.json')
    val_output = os.path.join(output_dir, 'annotations', 'instances_val.json')

    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, ensure_ascii=False, indent=2)

    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, ensure_ascii=False, indent=2)

    print(f"\n划分完成!")
    print(f"训练集标注文件: {train_output}")
    print(f"测试集标注文件: {val_output}")

    # 验证划分结果
    validate_split(train_annotations, val_annotations, coco_data)

    return train_output, val_output


def validate_split(train_data, val_data, original_data):
    """验证划分结果的正确性"""
    print("\n验证划分结果:")

    # 检查图像数量
    total_images = len(train_data['images']) + len(val_data['images'])
    print(f"总图像数量验证: {total_images} == {len(original_data['images'])}")

    # 检查标注数量
    total_annotations = len(train_data['annotations']) + len(val_data['annotations'])
    print(f"总标注数量验证: {total_annotations} == {len(original_data['annotations'])}")

    # 检查类别数量
    print(f"类别数量验证: {len(train_data['categories'])} == {len(original_data['categories'])}")

    # 检查是否有重叠
    train_image_ids = set(img['id'] for img in train_data['images'])
    val_image_ids = set(img['id'] for img in val_data['images'])
    overlap = train_image_ids & val_image_ids
    print(f"训练集-测试集重叠图像: {len(overlap)}")

    # 统计各类别在训练集和测试集中的分布
    print("\n各类别分布统计:")
    category_stats = defaultdict(lambda: {'train': 0, 'val': 0})

    for ann in train_data['annotations']:
        category_stats[ann['category_id']]['train'] += 1

    for ann in val_data['annotations']:
        category_stats[ann['category_id']]['val'] += 1

    for cat in original_data['categories']:
        cat_id = cat['id']
        cat_name = cat['name']
        train_count = category_stats[cat_id]['train']
        val_count = category_stats[cat_id]['val']
        total_count = train_count + val_count
        train_percent = (train_count / total_count * 100) if total_count > 0 else 0

        print(f"  {cat_name}({cat_id}): 训练集 {train_count}, 测试集 {val_count}, 训练占比 {train_percent:.1f}%")


def copy_images_by_annotation(image_dir, output_dir, train_annotation_file, val_annotation_file):
    """
    根据标注文件将图像复制到对应的训练集和测试集目录
    """
    # 创建图像目录
    train_image_dir = os.path.join(output_dir, 'train')
    val_image_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    # 读取训练集标注
    with open(train_annotation_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 读取测试集标注
    with open(val_annotation_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # 复制训练集图像
    print("\n复制训练集图像...")
    for image in train_data['images']:
        src_path = os.path.join(image_dir, image['file_name'])
        dst_path = os.path.join(train_image_dir, image['file_name'])

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 训练集图像不存在: {src_path}")

    # 复制测试集图像
    print("复制测试集图像...")
    for image in val_data['images']:
        src_path = os.path.join(image_dir, image['file_name'])
        dst_path = os.path.join(val_image_dir, image['file_name'])

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 测试集图像不存在: {src_path}")

    print(f"训练集图像复制到: {train_image_dir}")
    print(f"测试集图像复制到: {val_image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='划分COCO数据集为训练集和测试集')
    parser.add_argument('--coco_file', type=str, required=True, help='COCO标注文件路径')
    parser.add_argument('--output_dir', type=str, default='split_dataset', help='输出目录')
    parser.add_argument('--image_dir', type=str, help='图像文件目录（如果提供则会复制图像）')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 执行划分
    train_anno, val_anno = split_coco_dataset(
        args.coco_file,
        args.output_dir,
        args.train_ratio,
        args.seed
    )

    # 如果提供了图像目录，则复制图像
    if args.image_dir and os.path.exists(args.image_dir):
        copy_images_by_annotation(args.image_dir, args.output_dir, train_anno, val_anno)