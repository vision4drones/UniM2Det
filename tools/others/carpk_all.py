import os
import json
import glob


def convert_all_txt_to_coco(annotations_dir, output_file, image_dir=None):
    """
    将目录下所有TXT标注文件转换为单个COCO格式JSON文件

    Args:
        annotations_dir: 标注文件目录
        output_file: 输出JSON文件路径
        image_dir: 图像文件目录（用于获取图像尺寸，可选）
    """

    # 类别映射（根据您的要求，1对应car）
    categories = [{"id": 1, "name": "car", "supercategory": "vehicle"}]

    # COCO格式模板
    coco_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "All annotations converted to COCO format",
            "contributor": "",
            "url": "",
            "date_created": "2024-01-01"
        },
        "licenses": [{"id": 1, "name": "Unknown License", "url": ""}],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    # 查找所有TXT标注文件
    txt_files = glob.glob(os.path.join(annotations_dir, "*.txt"))
    print(f"找到 {len(txt_files)} 个TXT标注文件")

    # 排除划分文件
    exclude_files = ['train.txt', 'test.txt']
    txt_files = [f for f in txt_files if os.path.basename(f) not in exclude_files]
    print(f"处理后剩余 {len(txt_files)} 个标注文件")

    image_id = 0
    annotation_id = 0

    # 用于记录已处理的图像，避免重复
    processed_images = set()

    for txt_file in txt_files:
        file_name = os.path.basename(txt_file)
        print(f"处理: {file_name}")

        # 从文件名提取图像名称（去掉.txt扩展名）
        image_base_name = file_name.replace('.txt', '')

        # 如果图像已经处理过，跳过
        if image_base_name in processed_images:
            continue

        processed_images.add(image_base_name)

        # 添加图像信息
        image_info = {
            "id": image_id,
            "file_name": f"{image_base_name}.png",  # 假设图像格式为png
            "height": 0,  # 需要根据实际情况填写
            "width": 0  # 需要根据实际情况填写
        }

        # 如果提供了图像目录，尝试获取图像尺寸
        if image_dir:
            image_path = os.path.join(image_dir, f"{image_base_name}.png")
            if os.path.exists(image_path):
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        width, height = img.size
                    image_info["height"] = height
                    image_info["width"] = width
                except ImportError:
                    print("PIL未安装，无法自动获取图像尺寸")
                except Exception as e:
                    print(f"无法读取图像尺寸 {image_path}: {e}")

        coco_data["images"].append(image_info)

        # 读取并解析标注文件
        try:
            with open(txt_file, 'r') as f:
                annotations = f.readlines()

            for ann_line in annotations:
                ann_line = ann_line.strip()
                if not ann_line:
                    continue

                # 解析标注：x_min y_min x_max y_max class_id
                parts = ann_line.split()
                if len(parts) >= 5:
                    x_min = int(parts[0])
                    y_min = int(parts[1])
                    x_max = int(parts[2])
                    y_max = int(parts[3])
                    class_id = int(parts[4])

                    # 计算边界框的宽度和高度
                    width = x_max - x_min
                    height = y_max - y_min

                    # 确保宽度和高度为正数
                    if width > 0 and height > 0:
                        # 添加标注信息
                        annotation_info = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                            "segmentation": []  # COCO格式需要这个字段
                        }
                        coco_data["annotations"].append(annotation_info)
                        annotation_id += 1
                    else:
                        print(f"警告: 无效的边界框尺寸 in {file_name}: {ann_line}")
                else:
                    print(f"警告: 无效的标注格式 in {file_name}: {ann_line}")

        except Exception as e:
            print(f"错误: 处理文件 {txt_file} 时出错: {e}")
            continue

        image_id += 1

    # 保存COCO格式的JSON文件
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n转换完成!")
    print(f"输出文件: {output_file}")
    print(f"包含 {len(coco_data['images'])} 张图像")
    print(f"包含 {len(coco_data['annotations'])} 个标注")
    print(f"类别信息: {[cat['name'] for cat in coco_data['categories']]}")


def split_train_test_coco(full_coco_file, train_ratio=0.8, output_dir=None):
    """
    将完整的COCO标注文件划分为训练集和测试集

    Args:
        full_coco_file: 完整的COCO JSON文件路径
        train_ratio: 训练集比例
        output_dir: 输出目录
    """

    if output_dir is None:
        output_dir = os.path.dirname(full_coco_file)

    with open(full_coco_file, 'r') as f:
        full_data = json.load(f)

    # 获取所有图像ID
    image_ids = [img['id'] for img in full_data['images']]

    # 随机打乱图像ID
    import random
    random.seed(42)  # 设置随机种子以保证可重复性
    random.shuffle(image_ids)

    # 划分训练集和测试集
    split_idx = int(len(image_ids) * train_ratio)
    train_image_ids = set(image_ids[:split_idx])
    test_image_ids = set(image_ids[split_idx:])

    # 创建训练集数据
    train_data = {
        "info": full_data["info"],
        "licenses": full_data["licenses"],
        "categories": full_data["categories"],
        "images": [],
        "annotations": []
    }

    # 创建测试集数据
    test_data = {
        "info": full_data["info"],
        "licenses": full_data["licenses"],
        "categories": full_data["categories"],
        "images": [],
        "annotations": []
    }

    # 分配图像到训练集和测试集
    for image in full_data['images']:
        if image['id'] in train_image_ids:
            train_data['images'].append(image)
        else:
            test_data['images'].append(image)

    # 分配标注到训练集和测试集
    for annotation in full_data['annotations']:
        if annotation['image_id'] in train_image_ids:
            train_data['annotations'].append(annotation)
        else:
            test_data['annotations'].append(annotation)

    # 保存划分后的文件
    train_output = os.path.join(output_dir, "instances_train.json")
    test_output = os.path.join(output_dir, "instances_val.json")

    with open(train_output, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(test_output, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\n数据集划分完成!")
    print(f"训练集: {train_output} ({len(train_data['images'])} 张图像, {len(train_data['annotations'])} 个标注)")
    print(f"验证集: {test_output} ({len(test_data['images'])} 张图像, {len(test_data['annotations'])} 个标注)")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    annotations_dir = "/home/ldd/newdisk/CARpk/CARPK_devkit/data/Annotations"  # 标注文件所在目录
    output_file = "/home/ldd/newdisk/CARpk/CARPK_devkit/data/coco_annotations/annotations_all.json"  # 输出文件
    images_dir = "/home/ldd/newdisk/CARpk/CARPK_devkit/data/images"  # 图像文件目录（可选）

    # 第一步：转换所有标注文件为COCO格式
    convert_all_txt_to_coco(
        annotations_dir=annotations_dir,
        output_file=output_file,
        image_dir=images_dir
    )

    # 第二步：划分训练集和测试集（可选）
    # split_train_test_coco(
    #     full_coco_file=output_file,
    #     train_ratio=0.8,
    #     output_dir=os.path.dirname(output_file)
    # )