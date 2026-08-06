import pandas as pd
import numpy as np
import os
import sys
import argparse

INPUT_DIR = "./data_processing"
OUTPUT_DIR = "./processed_dataset"

SELECTED_FEATURES = [
    "Flow_Duration",
    "Tot_Fwd_Pkts",
    "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts",
    "TotLen_Bwd_Pkts",
    "Flow_Byts/s",
    "Fwd_Pkt_Len_Mean",
    "Bwd_Pkt_Len_Mean",
    "Pkt_Len_Mean"
]

CRITICAL_COLUMNS = SELECTED_FEATURES + ["Label"]


def clean_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].mean())

    df = df.dropna(subset=CRITICAL_COLUMNS)

    return df


def main(dataset_filename=None):
    print("=" * 60)
    print("数据清洗脚本")
    print("=" * 60)

    if not os.path.exists(INPUT_DIR):
        print(f"错误: 数据目录 {os.path.abspath(INPUT_DIR)} 不存在")
        print("请将数据集CSV文件放入 data_processing/ 目录")
        return False

    if dataset_filename:
        data_file = os.path.join(INPUT_DIR, dataset_filename)
        if not os.path.exists(data_file):
            print(f"错误: 未找到指定数据集文件 '{dataset_filename}'")
            print(f"请确认文件位于: {os.path.abspath(data_file)}")
            print(f"\n{INPUT_DIR} 目录中可用的CSV文件:")
            csv_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')])
            if csv_files:
                for f in csv_files:
                    print(f"  - {f}")
            else:
                print("  (无CSV文件)")
            return False
        print(f"\n使用指定数据集: {dataset_filename}")
    else:
        csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
        if not csv_files:
            print(f"错误: 在 {INPUT_DIR} 目录中未找到CSV文件")
            print("请将数据集CSV文件放入 data_processing/ 目录")
            return False
        if len(csv_files) > 1:
            print(f"警告: {INPUT_DIR} 中存在多个CSV文件，默认使用第一个: {csv_files[0]}")
            print("可通过 --dataset 参数指定具体文件名")
        data_file = os.path.join(INPUT_DIR, csv_files[0])

    print(f"\n1. 读取原始数据: {data_file}")
    df = pd.read_csv(data_file)
    print(f"原始数据: {df.shape[0]}行, {df.shape[1]}列")

    print("\n2. 查看原始标签分布...")
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        print(f"标签分布:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
    else:
        print("警告: 未找到Label列，无法进行标签编码")
        return False

    missing_cols = [c for c in CRITICAL_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"错误: 数据集缺少以下关键列: {missing_cols}")
        print("请确认数据集格式是否正确，或在 SELECTED_FEATURES 中调整特征列名")
        return False

    print("\n3. 数据清洗...")
    df_clean = clean_data(df)
    print(f"清洗后数据: {df_clean.shape[0]}行, {df_clean.shape[1]}列")

    if df_clean.shape[0] == 0:
        print("错误: 清洗后数据为空，可能是关键列中存在过多空值")
        return False

    print("\n4. 标签编码 (Normal→0, Anomaly→1)...")
    if 'Label' in df_clean.columns:
        df_clean['Label'] = df_clean['Label'].map({'Normal': 0, 'Anomaly': 1})
        label_counts = df_clean['Label'].value_counts()
        print(f"标签分布: 正常流量(0)={label_counts.get(0, 0)}, 恶意流量(1)={label_counts.get(1, 0)}")

    print("\n5. 保存清洗后数据...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "processed_dataset.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"  - 保存路径: {os.path.abspath(output_path)}")
    print(f"  - 数据量: {df_clean.shape[0]}行, {df_clean.shape[1]}列")

    print("\n" + "=" * 60)
    print("数据清洗完成！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据清洗脚本，支持指定数据集文件")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="指定要处理的数据集CSV文件名（位于data_processing/目录下）。不指定则使用默认的第一个CSV文件。")
    args = parser.parse_args()

    success = main(dataset_filename=args.dataset)
    sys.exit(0 if success else 1)