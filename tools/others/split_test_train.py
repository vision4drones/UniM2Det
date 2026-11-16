import os
import json
import shutil
from pathlib import Path


def split_coco_dataset(annotations_file, image_dir, output_dir,
                       train_txt, test_txt, val_txt=None,
                       image_extensions=['.jpg', '.png']):
    """
    根据txt文件划分COCO格式数据集

    Args:
        annotations_file: COCO标注文件路径
        image_dir: 原始图片目录
        output_dir: 输出目录
        train_txt: 训练集txt文件
        test_txt: 测试集txt文件
        val_txt: 验证集txt文件（可选）
        image_extensions: 图片后缀列表
    """

    # 创建输出目录
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    val_dir = output_dir / 'val' if val_txt else None

    for dir_path in [train_dir, test_dir, val_dir]:
        if dir_path:
            dir_path.mkdir(parents=True, exist_ok=True)

    # 读取COCO标注文件
    print(f"读取标注文件: {annotations_file}")
    with open(annotations_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 读取划分文件
    def read_split_file(txt_file):
        """读取划分文件中的图片名（不含后缀）"""
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"划分文件不存在: {txt_file}")

        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        image_names = []
        for line in lines:
            line = line.strip()
            if line:
                # 去除可能的后缀
                # line = Path(line).stem
                image_names.append(line)

        return set(image_names)

    print("读取划分文件...")
    train_names = read_split_file(train_txt)
    test_names = read_split_file(test_txt)
    val_names = read_split_file(val_txt) if val_txt else set()

    print(f"训练集图片数: {len(train_names)}")
    print(f"测试集图片数: {len(test_names)}")
    if val_txt:
        print(f"验证集图片数: {len(val_names)}")

    # 构建图片名到图片信息的映射
    image_name_to_info = {}
    for image_info in coco_data['images']:
        image_name = Path(image_info['file_name']).stem
        image_name_to_info[image_name] = image_info

    # 查找图片文件的实际路径
    def find_image_file(image_name):
        """根据图片名查找实际文件路径"""
        for ext in image_extensions:
            image_path = Path(image_dir) / f"{image_name}{ext}"
            if image_path.exists():
                return image_path
        return None

    # 划分数据集
    def process_split(split_names, output_subdir, split_name):
        """处理单个划分"""
        split_images = []
        split_annotations = []
        image_id_map = {}  # 旧图片ID到新图片ID的映射

        print(f"\n处理{split_name}集...")
        copied_count = 0
        missing_count = 0

        for image_name in split_names:
            # 查找图片文件
            image_path = find_image_file(image_name)
            if not image_path:
                print(f"警告: 未找到图片 {image_name}")
                missing_count += 1
                continue

            # 获取图片信息
            if image_name not in image_name_to_info:
                print(f"警告: 标注文件中未找到图片 {image_name}")
                missing_count += 1
                continue

            old_image_info = image_name_to_info[image_name]

            # 复制图片文件
            new_image_name = f"{image_name}{image_path.suffix}"
            new_image_path = output_subdir / new_image_name
            shutil.copy2(image_path, new_image_path)
            copied_count += 1

            # 更新图片信息
            new_image_id = len(split_images) + 1
            image_id_map[old_image_info['id']] = new_image_id

            new_image_info = old_image_info.copy()
            new_image_info['id'] = new_image_id
            new_image_info['file_name'] = new_image_name
            split_images.append(new_image_info)

        print(f"{split_name}集: 成功复制 {copied_count} 张图片, 缺失 {missing_count} 张")
        return split_images, split_annotations, image_id_map

    # 处理各个划分
    train_images, train_annotations, train_id_map = process_split(train_names, train_dir, "训练")
    test_images, test_annotations, test_id_map = process_split(test_names, test_dir, "测试")

    if val_txt:
        val_images, val_annotations, val_id_map = process_split(val_names, val_dir, "验证")

    # 处理标注信息
    def filter_annotations(original_annotations, image_id_map):
        """根据图片ID映射过滤和更新标注"""
        filtered_annotations = []

        for annotation in original_annotations:
            old_image_id = annotation['image_id']
            if old_image_id in image_id_map:
                new_annotation = annotation.copy()
                new_annotation['image_id'] = image_id_map[old_image_id]
                new_annotation['id'] = len(filtered_annotations) + 1
                filtered_annotations.append(new_annotation)

        return filtered_annotations

    print("\n处理标注信息...")
    all_id_map = {**train_id_map, **test_id_map, **(val_id_map if val_txt else {})}
    filtered_annotations = filter_annotations(coco_data['annotations'], all_id_map)

    # 保存划分后的标注文件
    def save_split_annotation(images, annotations, categories, output_path):
        """保存划分后的标注文件"""
        split_data = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)

        print(f"保存标注文件: {output_path}")

    # 保存各个划分的标注文件
    train_annotations = filter_annotations(coco_data['annotations'], train_id_map)
    test_annotations = filter_annotations(coco_data['annotations'], test_id_map)

    save_split_annotation(train_images, train_annotations, coco_data['categories'],
                          output_dir / 'train_annotations.json')
    save_split_annotation(test_images, test_annotations, coco_data['categories'],
                          output_dir / 'test_annotations.json')

    if val_txt:
        val_annotations = filter_annotations(coco_data['annotations'], val_id_map)
        save_split_annotation(val_images, val_annotations, coco_data['categories'],
                              output_dir / 'val_annotations.json')

    print(f"\n数据集划分完成！输出目录: {output_dir}")


def main():
    # 配置参数 - 请根据实际情况修改这些路径
    config = {
        'annotations_file': '/home/ldd/newdisk/State-Air/state_air_coco_clean.json',  # COCO标注文件
        'image_dir': '/home/ldd/newdisk/State-Air/JPEGImages',  # 原始图片目录
        'output_dir': '/home/ldd/newdisk/State-Air/',  # 输出目录
        'train_txt': '/home/ldd/newdisk/State-Air/ImageSets/Main/train.txt',  # 训练集划分文件
        'test_txt': '/home/ldd/newdisk/State-Air/ImageSets/Main/test.txt',  # 测试集划分文件
        'val_txt': '/home/ldd/newdisk/State-Air/ImageSets/Main/val.txt',  # 验证集划分文件（可选）
        'image_extensions': ['.jpg', '.png', '.jpeg']  # 支持的图片后缀
    }

    # 如果val.txt不存在，设置为None
    if not os.path.exists(config['val_txt']):
        print("未找到val.txt文件，将只划分训练集和测试集")
        config['val_txt'] = None

    try:
        split_coco_dataset(**config)
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()