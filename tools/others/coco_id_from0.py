# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict


def reset_category_ids_to_zero(input_json, output_json):
    """
    将COCO格式标注文件中的categories id改为从0开始

    Args:
        input_json: 输入的COCO格式JSON文件路径
        output_json: 输出的JSON文件路径
    """

    # 读取原始JSON文件
    with open(input_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 创建类别ID映射（旧ID -> 新ID）
    category_mapping = {}

    # 对类别按原始ID排序，然后重新编号从0开始
    sorted_categories = sorted(coco_data['categories'], key=lambda x: x['id'])

    new_categories = []
    for new_id, category in enumerate(sorted_categories):
        old_id = category['id']
        category_mapping[old_id] = new_id

        # 更新类别信息
        new_category = category.copy()
        new_category['id'] = new_id
        new_categories.append(new_category)

    # 更新annotations中的category_id
    new_annotations = []
    for annotation in coco_data['annotations']:
        old_category_id = annotation['category_id']
        if old_category_id in category_mapping:
            new_annotation = annotation.copy()
            new_annotation['category_id'] = category_mapping[old_category_id]
            new_annotations.append(new_annotation)
        else:
            print(f"警告: 标注 {annotation['id']} 的类别ID {old_category_id} 在类别列表中不存在")
            new_annotations.append(annotation)

    # 构建新的COCO数据
    new_coco_data = {
        "images": coco_data['images'],
        "annotations": new_annotations,
        "categories": new_categories
    }

    # 保存新的JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print(f"转换完成！")
    print(f"原始类别数量: {len(coco_data['categories'])}")
    print(f"新类别数量: {len(new_categories)}")
    print(f"标注数量: {len(new_annotations)}")
    print(f"类别ID映射:")
    for old_id, new_id in category_mapping.items():
        category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == old_id), "未知")
        print(f"  {old_id} -> {new_id} ({category_name})")
    print(f"输出文件: {output_json}")


def reset_category_ids_custom(input_json, output_json, start_id=0):
    """
    将COCO格式标注文件中的categories id改为从指定值开始

    Args:
        input_json: 输入的COCO格式JSON文件路径
        output_json: 输出的JSON文件路径
        start_id: 起始ID，默认为0
    """

    with open(input_json, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 创建类别ID映射
    category_mapping = {}

    # 对类别按原始ID排序，然后重新编号
    sorted_categories = sorted(coco_data['categories'], key=lambda x: x['id'])

    new_categories = []
    for i, category in enumerate(sorted_categories):
        old_id = category['id']
        new_id = start_id + i
        category_mapping[old_id] = new_id

        new_category = category.copy()
        new_category['id'] = new_id
        new_categories.append(new_category)

    # 更新annotations中的category_id
    new_annotations = []
    for annotation in coco_data['annotations']:
        old_category_id = annotation['category_id']
        if old_category_id in category_mapping:
            new_annotation = annotation.copy()
            new_annotation['category_id'] = category_mapping[old_category_id]
            new_annotations.append(new_annotation)
        else:
            print(f"警告: 标注 {annotation['id']} 的类别ID {old_category_id} 在类别列表中不存在")
            new_annotations.append(annotation)

    # 构建新的COCO数据
    new_coco_data = {
        "images": coco_data['images'],
        "annotations": new_annotations,
        "categories": new_categories
    }

    # 保存新的JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, indent=2, ensure_ascii=False)

    print(f"转换完成！")
    print(f"类别ID从 {start_id} 开始重新编号")
    print(f"处理了 {len(new_categories)} 个类别和 {len(new_annotations)} 个标注")
    print(f"输出文件: {output_json}")


def verify_category_ids(json_file):
    """
    验证COCO格式文件中类别ID的连续性

    Args:
        json_file: COCO格式JSON文件路径
    """

    with open(json_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    category_ids = [cat['id'] for cat in coco_data['categories']]
    min_id = min(category_ids) if category_ids else 0
    max_id = max(category_ids) if category_ids else 0
    expected_ids = set(range(min_id, max_id + 1))
    actual_ids = set(category_ids)

    missing_ids = expected_ids - actual_ids
    duplicate_ids = len(category_ids) - len(actual_ids)

    print(f"文件: {json_file}")
    print(f"类别数量: {len(category_ids)}")
    print(f"最小ID: {min_id}")
    print(f"最大ID: {max_id}")
    print(f"缺失的ID: {sorted(missing_ids) if missing_ids else '无'}")
    print(f"重复的ID数量: {duplicate_ids}")
    print(f"ID是否连续: {'是' if not missing_ids and duplicate_ids == 0 else '否'}")


def main():
    parser = argparse.ArgumentParser(description='修改COCO格式标注文件中categories的id')
    parser.add_argument('--input_json', type=str, required=True, help='输入的COCO格式JSON文件路径')
    parser.add_argument('--output_json', type=str, required=True, help='输出的JSON文件路径')
    parser.add_argument('--start_id', type=int, default=0, help='起始ID，默认为0')
    parser.add_argument('--verify', action='store_true', help='仅验证文件而不修改')

    args = parser.parse_args()

    if args.verify:
        verify_category_ids(args.input_json)
    else:
        reset_category_ids_custom(
            input_json=args.input_json,
            output_json=args.output_json,
            start_id=args.start_id
        )


if __name__ == "__main__":
    # 使用示例
    # 直接从0开始
    # reset_category_ids_to_zero("coco_annotations.json", "coco_annotations_zero.json")

    # 自定义起始ID
    # reset_category_ids_custom("coco_annotations.json", "coco_annotations_new.json", start_id=0)

    # 验证文件
    # verify_category_ids("coco_annotations.json")

    main()