# coding: utf-8
import cv2
import mmcv
import numpy as np
import os
import torch
from mmdet.apis import inference_detector_heatmap, init_detector_heatmap
from argparse import ArgumentParser
import traceback


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    return heatmap


def draw_feature_map(model, img_path, save_dir):
    """
    处理单张图像，将各层热力图保存至统一目录
    文件命名: {原图文件名}_cam_P{层索引}.png
    """
    img = mmcv.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 提取安全文件名前缀（不含扩展名）
    img_stem = os.path.splitext(os.path.basename(img_path))[0]

    model.eval()
    model.draw_heatmap = True
    featuremaps = inference_detector_heatmap(model, img)

    saved_count = 0
    for idx, featuremap in enumerate(featuremaps, start=2):
        try:
            heatmap = featuremap_2_heatmap(featuremap)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)

            # 生成唯一文件名
            save_path = os.path.join(save_dir, f"{img_stem}_cam_P{idx}.png")
            cv2.imwrite(save_path, superimposed_img)
            saved_count += 1
        except Exception as e:
            print(f"  ⚠ 跳过层 P{idx}: {str(e)}")
            continue
    return saved_count


def main():
    parser = ArgumentParser(description="批量生成目标检测模型热力图（统一目录保存）")
    parser.add_argument('image_folder', help='输入图像文件夹路径')
    parser.add_argument('save_dir', help='热力图统一保存目录')
    parser.add_argument('config', help='模型配置文件路径')
    parser.add_argument('checkpoint', help='模型权重文件路径')
    parser.add_argument('--device', default='cuda:0', help='推理设备 (默认: cuda:0)')
    parser.add_argument('--ext', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'],
                        help='要处理的图像扩展名列表 (默认: 常见格式)')
    args = parser.parse_args()

    # 验证并创建保存目录
    if not os.path.isdir(args.image_folder):
        raise ValueError(f"❌ 图像文件夹不存在: {args.image_folder}")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"📁 热力图将保存至: {os.path.abspath(args.save_dir)}\n")

    # 收集有效图像
    valid_exts = set(ext.lower() for ext in args.ext)
    img_files = sorted([
        f for f in os.listdir(args.image_folder)
        if os.path.isfile(os.path.join(args.image_folder, f))
           and os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not img_files:
        raise ValueError(f"❌ 未在 {args.image_folder} 中找到支持的图像文件 (支持: {valid_exts})")
    print(f"✅ 找到 {len(img_files)} 张图像，支持扩展名: {valid_exts}\n")

    # 加载模型
    print("⚙️  加载模型...")
    model = init_detector_heatmap(args.config, args.checkpoint, device=args.device)
    print("✅ 模型加载成功\n")

    # 批量处理
    total_saved = 0
    success_imgs = 0
    for idx, img_file in enumerate(img_files, 1):
        img_path = os.path.join(args.image_folder, img_file)
        img_stem = os.path.splitext(img_file)[0]
        print(f"[{idx}/{len(img_files)}] 处理: {img_file}")

        try:
            saved = draw_feature_map(model, img_path, args.save_dir)
            if saved > 0:
                print(f"  ✅ 保存 {saved} 个热力图: {img_stem}_cam_P*.png")
                total_saved += saved
                success_imgs += 1
            else:
                print(f"  ⚠  未生成有效热力图")
        except Exception as e:
            print(f"  ❌ 处理失败: {str(e)}")
            traceback.print_exc()
            continue

    # 汇总结果
    print("\n" + "=" * 60)
    print(f"🎉 处理完成!")
    print(f"   - 成功处理图像: {success_imgs}/{len(img_files)}")
    print(f"   - 生成热力图总数: {total_saved}")
    print(f"   - 保存目录: {os.path.abspath(args.save_dir)}")
    print(f"   - 命名规则: {{原图名}}_cam_P{{层索引}}.png (例: cat_cam_P3.png)")
    print("=" * 60)


if __name__ == '__main__':
    main()