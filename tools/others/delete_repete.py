import json

# 读取标注文件
annotation_file = '/home/ldd/newdisk/CARpk/CARPK_devkit/data/coco_annotations/instances_train.json'
with open(annotation_file, 'r') as f:
    data = json.load(f)

# 检查重复的 annotation ID
annotations = data['annotations']
print(f"总标注数量: {len(annotations)}")

# 查找重复的 ID
id_set = set()
duplicate_ids = []
for ann in annotations:
    if ann['id'] in id_set:
        duplicate_ids.append(ann['id'])
    else:
        id_set.add(ann['id'])

print(f"重复的 ID 数量: {len(duplicate_ids)}")
print(f"重复的 ID: {duplicate_ids[:10]}")  # 只显示前10个

if duplicate_ids:
    # 修复重复 ID
    print("开始修复重复 ID...")
    new_id = max(id_set) + 1
    for ann in annotations:
        if ann['id'] in duplicate_ids:
            ann['id'] = new_id
            new_id += 1

    # 保存修复后的文件
    output_file = annotation_file.replace('.json', '_fixed.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"修复完成，保存为: {output_file}")
else:
    print("没有发现重复 ID")