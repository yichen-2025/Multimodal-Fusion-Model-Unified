import os
import json
import pandas as pd
from datetime import datetime

BASE_LOG_DIR = "./logs"


def check_gpu_available():
    """
    检查是否有可用的GPU设备
    
    如果没有GPU，则发出警告并要求用户确认是否继续使用CPU运行。
    在需要GPU的程序（如模型训练、推理、BERT嵌入提取）执行前必须调用。
    """
    import torch
    
    if not torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("⚠️  WARNING: 未检测到可用的GPU设备！")
        print("=" * 60)
        print("本程序需要GPU来高效运行深度学习模型（LLM、BERT等）。")
        print("当前环境中未检测到CUDA兼容的GPU，将使用CPU运行，")
        print("这可能导致程序运行速度极慢，甚至可能因内存不足而崩溃。")
        print("=" * 60)
        
        while True:
            try:
                choice = input("\n是否继续使用CPU运行？(y/N): ").strip().lower()
                if choice == 'y':
                    print("\n警告：用户确认使用CPU继续运行。")
                    print("注意：运行时间可能非常长。建议在具备GPU的环境中运行此程序。")
                    return
                else:
                    print("\n程序已取消。请在配备GPU的环境中重新运行。")
                    print("如需强制在CPU上运行，请在代码中手动跳过此检查。")
                    exit(1)
            except (KeyboardInterrupt, EOFError):
                print("\n\n程序已中断。")
                exit(1)
    
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
    
    print(f"[GPU检查] 检测到 {gpu_count} 个GPU设备:")
    print(f"  - GPU 0: {gpu_name}")
    print(f"  - 显存: {gpu_memory:.1f} GB")
    print(f"  - CUDA可用: ✓")


def get_log_dir(log_type):
    """获取指定类型的日志目录"""
    log_dir = os.path.join(BASE_LOG_DIR, log_type)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_next_log_id(log_type):
    """获取下一个可用的日志ID（自动递增）"""
    log_dir = get_log_dir(log_type)
    
    max_id = -1
    for f in os.listdir(log_dir):
        if f.startswith("log_") and f.endswith(".json"):
            try:
                idx = int(f.replace("log_", "").replace(".json", ""))
                if idx > max_id:
                    max_id = idx
            except ValueError:
                pass
    
    return max_id + 1


def save_log(log_type, log_data):
    """
    保存日志文件
    
    功能：将日志数据保存为JSON文件，同时更新索引CSV文件
    
    Args:
        log_type (str): 日志类型，可选值：
            - 'subset'   : 子集提取日志
            - 'split'    : 数据集划分日志
            - 'training' : 模型训练日志
            - 'testing'  : 模型测试日志
        log_data (dict): 日志数据字典
        
    Returns:
        int: 日志ID
    """
    log_dir = get_log_dir(log_type)
    log_id = get_next_log_id(log_type)
    
    log_data['log_id'] = log_id
    log_data['timestamp'] = datetime.now().isoformat()
    
    json_path = os.path.join(log_dir, f"log_{log_id}.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    index_file = os.path.join(log_dir, "index.csv")
    
    index_columns = {
        'subset': ['log_id', 'timestamp', 'dataset_id', 'total_samples', 'positive_samples', 'negative_samples'],
        'split': ['log_id', 'timestamp', 'dataset_id', 'split_id', 'train_samples', 'test_samples'],
        'training': ['log_id', 'timestamp', 'model_id', 'dataset_id', 'split_id', 'learning_rate', 'epochs', 'duration_seconds'],
        'testing': ['log_id', 'timestamp', 'model_id', 'dataset_id', 'split_id', 'accuracy', 'f1', 'duration_seconds']
    }
    
    csv_row = {}
    for col in index_columns.get(log_type, []):
        csv_row[col] = log_data.get(col, '')
    
    if os.path.exists(index_file):
        df = pd.read_csv(index_file)
        df = pd.concat([df, pd.DataFrame([csv_row])], ignore_index=True)
        df.to_csv(index_file, index=False)
    
    return log_id