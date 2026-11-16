import json
import numpy as np
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
            x_coords = coords[0::2]
            y_coords = coords[1::2]

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


def get_npy_image_info(npy_path):
    """
    获取npy格式图像的尺寸和元数据信息
    """
    try:
        # 方法1: 使用mmap模式读取shape而不加载全部数据（推荐用于大文件）
        img_data = np.load(npy_path, mmap_mode='r')

        # npy文件的shape通常是 (H, W) 或 (H, W, C)
        if len(img_data.shape) == 2:
            height, width = img_data.shape
            channels = 1
        elif len(img_data.shape) == 3:
            channels, width, height = img_data.shape
        else:
            raise ValueError(f"不支持的shape: {img_data.shape}")

        return channels, width, height
    except Exception as e:
        print(f"错误: 无法读取npy文件 {npy_path}: {e}")
        return None, None, None


def convert_npy_to_coco(txt_dir, npy_dir, output_json_path, category_mapping=None):
    """
    将npy多光谱图像的txt标注转换为COCO格式

    参数:
        txt_dir: 包含txt标注文件的目录
        npy_dir: 包含npy图像文件的目录
        output_json_path: 输出的COCO JSON文件路径
        category_mapping: 类别映射字典
    """

    # 获取所有txt文件
    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files:
        print(f"错误: 在 {txt_dir} 中没有找到txt文件")
        return None

    print(f"找到 {len(txt_files)} 个标注文件")

    # 第一遍：收集所有类别
    all_categories = set()
    file_annotations = {}

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
            "description": "Converted from multi-spectral npy HBB format to COCO",
            "version": "1.0",
            "year": 2024,
            "contributor": "Conversion Script",
            "date_created": "2024/01/01",
            "format": "npy_multi_spectral"
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
    skipped_images = []

    for img_name, annotations in tqdm(file_annotations.items(), desc="转换标注"):
        # 查找npy文件
        npy_path = Path(npy_dir) / f"{img_name}.npy"

        if not npy_path.exists():
            print(f"警告: 找不到npy文件 {npy_path}，跳过...")
            skipped_images.append(img_name)
            continue

        # 获取图像尺寸
        channels, width, height = get_npy_image_info(npy_path)

        if height is None:
            print(f"警告: 无法读取npy文件 {npy_path}，跳过...")
            skipped_images.append(img_name)
            continue

        # 添加图像信息（包含多光谱特有信息）
        coco_data["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"{img_name}.npy",
            "metadata": {  # 添加多光谱图像元数据
                "format": "npy",
                "channels": channels,
                "dtype": str(np.load(npy_path, mmap_mode='r').dtype)
            }
        })

        # 添加该图像的所有标注
        for ann in annotations:
            x_min, y_min, x_max, y_max = ann['bbox']

            # 计算COCO格式的bbox
            width_box = x_max - x_min
            height_box = y_max - y_min

            # 验证bbox有效性
            if width_box <= 0 or height_box <= 0:
                print(f"警告: 图像 {img_name} 中有无效bbox，跳过...")
                continue

            # 确保bbox在图像范围内
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            width_box = min(width_box, width - x_min)
            height_box = min(height_box, height - y_min)

            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_mapping[ann['category']],
                "bbox": [float(x_min), float(y_min), float(width_box), float(height_box)],
                "area": float(width_box * height_box),
                "segmentation": [],
                "iscrowd": 0,
                "attributes": {
                    "difficult": ann['difficult']
                }
            }

            coco_data["annotations"].append(coco_annotation)
            annotation_id += 1

        image_id += 1

    # 保存COCO JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("转换完成！")
    print(f"输出文件: {output_json_path}")
    print(f"成功转换图像: {len(coco_data['images'])} 张")
    print(f"跳过的图像: {len(skipped_images)} 张")
    print(f"标注数量: {len(coco_data['annotations'])} 个")
    print(f"类别数量: {len(coco_data['categories'])} 个")

    if skipped_images:
        print(f"\n跳过的图像列表: {skipped_images[:10]}")  # 只显示前10个

    # 按类别统计
    category_stats = {}
    for ann in coco_data['annotations']:
        cat_name = [c['name'] for c in coco_data['categories'] if c['id'] == ann['category_id']][0]
        category_stats[cat_name] = category_stats.get(cat_name, 0) + 1

    print("\n各类别标注统计:")
    for cat_name, count in sorted(category_stats.items()):
        print(f"  {cat_name}: {count} 个标注")
    print("=" * 60)

    return coco_data


def visualize_npy_bbox(npy_path, coco_json_path, image_id=1):
    """
    可视化npy图像和对应的bbox（可选功能）
    需要安装: pip install matplotlib opencv-python
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 加载COCO标注
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 获取图像信息
    img_info = None
    for img in coco_data['images']:
        if img['id'] == image_id:
            img_info = img
            break

    if img_info is None:
        print(f"找不到图像ID {image_id}")
        return

    # 加载npy图像
    npy_path = Path(npy_path) / img_info['file_name']
    img_data = np.load(npy_path)

    # 归一化显示（多光谱图像通常需要归一化）
    if len(img_data.shape) == 2:
        # 单通道
        img_display = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        plt.imshow(img_display, cmap='gray')
    elif len(img_data.shape) == 3:
        # 多通道，取前3个通道或平均值
        if img_data.shape[2] >= 3:
            img_display = img_data[:, :, :3]
        else:
            img_display = np.mean(img_data, axis=2)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        plt.imshow(img_display)

    # 绘制bbox
    ax = plt.gca()
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            x, y, w, h = ann['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            # 添加类别标签
            cat_name = [c['name'] for c in coco_data['categories'] if c['id'] == ann['category_id']][0]
            plt.text(x, y - 5, cat_name, color='red', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title(f"Image: {img_info['file_name']} (Channels: {img_info['metadata']['channels']})")
    plt.axis('off')
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 设置路径
    txt_directory = "/home/ldd/newdisk/MODA/train/labels/"  # txt标注目录
    npy_directory = "/home/ldd/newdisk/MODA/train/images/"  # npy图像目录
    output_json = "/home/ldd/newdisk/MODA/coco_anno/train.json"  # 输出文件

    # 转换
    convert_npy_to_coco(
        txt_dir=txt_directory,
        npy_dir=npy_directory,
        output_json_path=output_json
    )

    # 可选：可视化验证（需要matplotlib）
    # visualize_npy_bbox(npy_directory, output_json, image_id=1)