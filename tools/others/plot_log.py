import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log_file(log_path):
    """解析日志文件，提取训练指标"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 训练指标正则表达式
    epoch_pattern = r'Epoch\(train\)\s+\[(\d+)\]\[(\s*\d+)/\d+\].*?grad_norm:\s+([\d\.naninf]+)\s+loss:\s+([\d\.]+)'
    train_matches = re.findall(epoch_pattern, content)

    # 验证指标正则表达式 - 提取所有数据集（包含state_air）
    val_pattern = r'Epoch\(val\)\s+\[(\d+)\]\[\s*\d+/\d+\].*?' \
                  r'DroneVehicle/coco/bbox_mAP:\s+([\d\.-]+).*?' \
                  r'DroneVehicle/coco/bbox_mAP_50:\s+([\d\.-]+).*?' \
                  r'visdrone/coco/bbox_mAP:\s+([\d\.-]+).*?' \
                  r'visdrone/coco/bbox_mAP_50:\s+([\d\.-]+).*?' \
                  r'carPK/coco/bbox_mAP:\s+([\d\.-]+).*?' \
                  r'carPK/coco/bbox_mAP_50:\s+([\d\.-]+).*?' \
                  r'RGBT-tiny/coco/bbox_mAP:\s+([\d\.-]+).*?' \
                  r'RGBT-tiny/coco/bbox_mAP_50:\s+([\d\.-]+).*?' \
                  r'RTDOD/coco/bbox_mAP:\s+([\d\.-]+).*?' \
                  r'RTDOD/coco/bbox_mAP_50:\s+([\d\.-]+).*?' \
                  r'UAVDT/coco/bbox_mAP:\s+([\d\.-]+).*?' \
                  r'UAVDT/coco/bbox_mAP_50:\s+([\d\.-]+).*?' \
                  r'state_air/coco/bbox_mAP:\s+([\d\.-]+).*?' \
                  r'state_air/coco/bbox_mAP_50:\s+([\d\.-]+)'

    val_matches = re.findall(val_pattern, content, re.DOTALL)

    # 解析训练数据 - 重新计算全局步数
    epochs = []
    global_steps = []  # 改为全局步数
    grad_norms = []
    losses = []

    total_steps_per_epoch = 2244  # 根据日志中的 [step/2244]
    global_step = 0

    for epoch, step, grad_norm, loss in train_matches:
        epoch = int(epoch)
        step = int(step)

        # 计算全局步数: (epoch-1) * steps_per_epoch + current_step
        global_step = (epoch - 1) * total_steps_per_epoch + step

        epochs.append(epoch)
        global_steps.append(global_step)

        if grad_norm.lower() in ['nan', 'inf']:
            grad_norms.append(np.nan)
        else:
            grad_norms.append(float(grad_norm))

        losses.append(float(loss))

    # 解析验证数据
    val_epochs = []
    dataset_metrics = {
        'DroneVehicle': {'mAP': [], 'mAP_50': []},
        'visdrone': {'mAP': [], 'mAP_50': []},
        'carPK': {'mAP': [], 'mAP_50': []},
        'RGBT-tiny': {'mAP': [], 'mAP_50': []},
        'RTDOD': {'mAP': [], 'mAP_50': []},
        'UAVDT': {'mAP': [], 'mAP_50': []},
        'state_air': {'mAP': [], 'mAP_50': []}
    }

    for match in val_matches:
        epoch = int(match[0])
        val_epochs.append(epoch)

        # 提取每个数据集的指标（现在包含state_air）
        datasets = [
            'DroneVehicle', 'visdrone', 'carPK', 'RGBT-tiny', 'RTDOD', 'UAVDT', 'state_air'
        ]

        for i, dataset in enumerate(datasets):
            mAP_idx = 1 + i * 2
            mAP_50_idx = 2 + i * 2

            mAP_val = float(match[mAP_idx]) if match[mAP_idx] != '-1.0000' else np.nan
            mAP_50_val = float(match[mAP_50_idx]) if match[mAP_50_idx] != '-1.0000' else np.nan

            dataset_metrics[dataset]['mAP'].append(mAP_val)
            dataset_metrics[dataset]['mAP_50'].append(mAP_50_val)

    return {
        'epochs': epochs,
        'global_steps': global_steps,  # 返回全局步数
        'grad_norms': grad_norms,
        'losses': losses,
        'val_epochs': val_epochs,
        'dataset_metrics': dataset_metrics,
        'total_steps_per_epoch': total_steps_per_epoch
    }


def create_plots(data, selected_datasets=None):
    """创建可视化图表"""
    if selected_datasets is None:
        selected_datasets = list(data['dataset_metrics'].keys())

    # 创建更大的画布
    fig = plt.figure(figsize=(20, 16))

    # 定义子图布局
    gs = plt.GridSpec(3, 3, figure=fig)

    # 1. 损失变化曲线 - 使用全局步数
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['global_steps'], data['losses'], 'b-', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Global Training Steps')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Loss Over Global Steps')
    ax1.grid(True, alpha=0.3)

    # 添加epoch分界线 - 使用全局步数
    unique_epochs = sorted(set(data['epochs']))
    for epoch in unique_epochs:
        # 每个epoch开始时的全局步数
        epoch_start_step = (epoch - 1) * data['total_steps_per_epoch']
        ax1.axvline(x=epoch_start_step, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        # 在每个epoch开始处添加标签
        ax1.text(epoch_start_step, ax1.get_ylim()[1], f'E{epoch}',
                 rotation=90, verticalalignment='top', fontsize=8)

    # 2. 梯度范数变化 - 使用全局步数
    ax2 = fig.add_subplot(gs[0, 1])
    valid_grad_indices = [i for i, gn in enumerate(data['grad_norms']) if not np.isnan(gn)]
    valid_global_steps = [data['global_steps'][i] for i in valid_grad_indices]
    valid_grads = [data['grad_norms'][i] for i in valid_grad_indices]

    ax2.plot(valid_global_steps, valid_grads, 'g-', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Global Training Steps')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm Over Global Steps')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 添加epoch分界线
    for epoch in unique_epochs:
        epoch_start_step = (epoch - 1) * data['total_steps_per_epoch']
        ax2.axvline(x=epoch_start_step, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        ax2.text(epoch_start_step, ax2.get_ylim()[1], f'E{epoch}',
                 rotation=90, verticalalignment='top', fontsize=8)

    # 3. 每个epoch的平均损失
    ax3 = fig.add_subplot(gs[0, 2])
    epoch_avg_losses = {}
    for epoch, global_step, loss in zip(data['epochs'], data['global_steps'], data['losses']):
        if epoch not in epoch_avg_losses:
            epoch_avg_losses[epoch] = []
        epoch_avg_losses[epoch].append(loss)

    epoch_nums = sorted(epoch_avg_losses.keys())
    avg_losses = [np.mean(epoch_avg_losses[epoch]) for epoch in epoch_nums]

    ax3.bar(epoch_nums, avg_losses, alpha=0.7, color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Loss')
    ax3.set_title('Average Loss per Epoch')
    ax3.grid(True, alpha=0.3)

    # 4. 所有数据集的mAP变化（只显示选中的数据集）
    ax4 = fig.add_subplot(gs[1, :])
    colors = plt.cm.Set3(np.linspace(0, 1, len(selected_datasets)))

    for i, dataset in enumerate(selected_datasets):
        if dataset in data['dataset_metrics'] and data['val_epochs'] and data['dataset_metrics'][dataset]['mAP']:
            # 过滤掉NaN值
            valid_epochs = [epoch for epoch, mAP in zip(data['val_epochs'], data['dataset_metrics'][dataset]['mAP']) if
                            not np.isnan(mAP)]
            valid_mAP = [mAP for mAP in data['dataset_metrics'][dataset]['mAP'] if not np.isnan(mAP)]

            if valid_epochs:
                ax4.plot(valid_epochs, valid_mAP, 'o-', color=colors[i],
                         label=f'{dataset} mAP', linewidth=2, markersize=4)

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('mAP')
    ax4.set_title('Selected Datasets mAP Over Epochs')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)

    # 5. 所有数据集的mAP@50变化（只显示选中的数据集）
    ax5 = fig.add_subplot(gs[2, :])

    for i, dataset in enumerate(selected_datasets):
        if dataset in data['dataset_metrics'] and data['val_epochs'] and data['dataset_metrics'][dataset]['mAP_50']:
            # 过滤掉NaN值
            valid_epochs = [epoch for epoch, mAP in zip(data['val_epochs'], data['dataset_metrics'][dataset]['mAP_50'])
                            if not np.isnan(mAP)]
            valid_mAP_50 = [mAP for mAP in data['dataset_metrics'][dataset]['mAP_50'] if not np.isnan(mAP)]

            if valid_epochs:
                ax5.plot(valid_epochs, valid_mAP_50, 's-', color=colors[i],
                         label=f'{dataset} mAP@50', linewidth=2, markersize=4)

    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('mAP@50')
    ax5.set_title('Selected Datasets mAP@50 Over Epochs')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_analysis_selected_datasets.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_individual_dataset_plots(data, selected_datasets=None):
    """为每个数据集创建单独的详细图表"""
    if not data['val_epochs']:
        return

    if selected_datasets is None:
        selected_datasets = list(data['dataset_metrics'].keys())

    # 创建每个数据集的详细图表
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 调整为2x4布局以容纳7个数据集
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i, dataset in enumerate(selected_datasets):
        if i >= len(axes):
            break

        if dataset not in data['dataset_metrics']:
            continue

        metrics = data['dataset_metrics'][dataset]
        ax = axes[i]

        # 过滤NaN值
        valid_epochs_mAP = [epoch for epoch, mAP in zip(data['val_epochs'], metrics['mAP']) if not np.isnan(mAP)]
        valid_mAP = [mAP for mAP in metrics['mAP'] if not np.isnan(mAP)]

        valid_epochs_mAP50 = [epoch for epoch, mAP in zip(data['val_epochs'], metrics['mAP_50']) if not np.isnan(mAP)]
        valid_mAP_50 = [mAP for mAP in metrics['mAP_50'] if not np.isnan(mAP)]

        if valid_epochs_mAP:
            ax.plot(valid_epochs_mAP, valid_mAP, 'o-', color=colors[i % len(colors)],
                    label='mAP', linewidth=2, markersize=4)

        if valid_epochs_mAP50:
            ax.plot(valid_epochs_mAP50, valid_mAP_50, 's-', color=colors[i % len(colors)],
                    label='mAP@50', linewidth=2, markersize=4, alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('AP Score')
        ax.set_title(f'{dataset} Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(selected_datasets), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('individual_datasets_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    log_path = '/media/ldd/AE34-B0B9/atr/base.log'  # 请确保文件路径正确

    if not Path(log_path).exists():
        print(f"日志文件 {log_path} 不存在，请检查文件路径")
        return

    # 可用的数据集列表
    available_datasets = ['DroneVehicle', 'visdrone', 'carPK', 'RGBT-tiny', 'RTDOD', 'UAVDT', 'state_air']

    print("可用的数据集:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")

    # 让用户选择要绘制的数据集
    print("\n请选择要绘制的数据集（输入数字，多个用逗号分隔，或输入'all'选择所有）:")
    user_input = input().strip()

    if user_input.lower() == 'all':
        selected_datasets = available_datasets
    else:
        try:
            selected_indices = [int(x.strip()) - 1 for x in user_input.split(',')]
            selected_datasets = [available_datasets[i] for i in selected_indices if 0 <= i < len(available_datasets)]
        except ValueError:
            print("输入格式错误，将使用所有数据集")
            selected_datasets = available_datasets

    print(f"选择的数据集: {selected_datasets}")

    print("正在解析日志文件...")
    data = parse_log_file(log_path)

    print("正在生成综合图表...")
    create_plots(data, selected_datasets)

    print("正在生成各数据集详细图表...")
    create_individual_dataset_plots(data, selected_datasets)

    print("分析完成！")
    print("图表已保存为:")
    print("- 'training_analysis_selected_datasets.png' (综合图表)")
    print("- 'individual_datasets_performance.png' (各数据集详细图表)")


if __name__ == "__main__":
    main()