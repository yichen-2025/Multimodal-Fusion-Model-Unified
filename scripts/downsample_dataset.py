import pandas as pd
import os
import argparse

INPUT_FILE = "data_processing/IoT Network Intrusion Dataset.csv"
OUTPUT_FILE = "data_processing/IoT Network Intrusion Dataset_subset.csv"
TARGET_TOTAL = 20000
RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser(description="数据集下采样脚本")
    parser.add_argument("--input", "-i", default=INPUT_FILE, help="输入文件路径")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE, help="输出文件路径")
    parser.add_argument("--total", "-n", type=int, default=TARGET_TOTAL, help="目标样本总数")
    parser.add_argument("--ratio", "-r", type=float, default=0.5, help="正样本比例(默认0.5即均衡)")
    parser.add_argument("--seed", "-s", type=int, default=RANDOM_STATE, help="随机种子")
    args = parser.parse_args()

    print("=" * 60)
    print("数据集下采样脚本")
    print("=" * 60)

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return False

    input_size_mb = os.path.getsize(args.input) / 1024 / 1024
    print(f"\n输入文件: {args.input}")
    print(f"文件大小: {input_size_mb:.2f} MB")

    print(f"\n正在读取数据...")
    df = pd.read_csv(args.input)
    print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

    if 'Label' not in df.columns:
        print("错误: 数据集缺少 'Label' 列")
        return False

    print(f"\n原始标签分布:")
    label_counts = df['Label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

    normal_count = label_counts.get('Normal', 0)
    anomaly_count = label_counts.get('Anomaly', 0)

    if normal_count == 0 or anomaly_count == 0:
        print("错误: 缺少 Normal 或 Anomaly 类别的样本")
        return False

    target_normal = int(args.total * args.ratio)
    target_anomaly = args.total - target_normal

    print(f"\n目标样本数: {args.total} (Normal={target_normal}, Anomaly={target_anomaly})")

    if target_normal > normal_count:
        print(f"警告: Normal 样本不足 (需要{target_normal}, 仅有{normal_count})")
        target_normal = normal_count
        target_anomaly = min(args.total - target_normal, anomaly_count)

    if target_anomaly > anomaly_count:
        print(f"警告: Anomaly 样本不足 (需要{target_anomaly}, 仅有{anomaly_count})")
        target_anomaly = anomaly_count
        target_normal = args.total - target_anomaly

    print(f"\n实际采样数: Normal={target_normal}, Anomaly={target_anomaly}")

    normal_df = df[df['Label'] == 'Normal'].sample(n=target_normal, random_state=args.seed)
    anomaly_df = df[df['Label'] == 'Anomaly'].sample(n=target_anomaly, random_state=args.seed)

    subset = pd.concat([normal_df, anomaly_df], ignore_index=True)
    subset = subset.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\n子集数据: {subset.shape[0]} 行, {subset.shape[1]} 列")
    print(f"子集标签分布:")
    for label, count in subset['Label'].value_counts().items():
        print(f"  {label}: {count} ({count/len(subset)*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    subset.to_csv(args.output, index=False)

    output_size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\n输出文件: {args.output}")
    print(f"文件大小: {output_size_mb:.2f} MB")

    if output_size_mb > 100:
        print("⚠️  警告: 文件大小仍超过100MB，可能无法直接上传GitHub")
    else:
        print("✅ 文件大小符合GitHub 100MB限制")

    print("\n" + "=" * 60)
    print("下采样完成!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)