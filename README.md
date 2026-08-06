# 多模态融合网络流量分类模型

基于多模态融合的网络流量恶意检测模型，结合统计特征模态和文本描述模态进行流量分类。

## 项目简介

本项目实现了一个多模态融合模型，用于网络流量恶意检测。模型将网络流量数据转换为两种模态：

1. **统计特征模态**：9维数值特征（如流持续时间、包数量、字节速率等）
2. **文本描述模态**：768维BERT语义嵌入

通过多模态融合技术，将两种模态的特征进行融合，实现更准确的流量分类。

## 当前使用的数据集

**IoT Network Intrusion Dataset**：包含83个特征列，标签为 `Normal`（正常流量）和 `Anomaly`（恶意流量）。

### 选用的9个连续特征（统计特征模态）

| 特征列名 | 含义 |
|---------|------|
| `Flow_Duration` | 流持续时间 |
| `Tot_Fwd_Pkts` | 前向包数量 |
| `Tot_Bwd_Pkts` | 反向包数量 |
| `TotLen_Fwd_Pkts` | 前向包总长度 |
| `TotLen_Bwd_Pkts` | 反向包总长度 |
| `Flow_Byts/s` | 流字节速率 |
| `Fwd_Pkt_Len_Mean` | 前向包平均长度 |
| `Bwd_Pkt_Len_Mean` | 反向包平均长度 |
| `Pkt_Len_Mean` | 包平均长度 |

### 离散特征（文本描述模态）

利用 `Protocol`、`Src_IP`、`Src_Port`、`Dst_IP`、`Dst_Port` 等离散字段生成中文文本描述。

## 目录结构

```
多模态融合3/
├── main.py                    # 主入口文件（推荐运行方式）
├── scripts/                   # 脚本模块
│   ├── data_cleaning.py       # 数据清洗脚本
│   ├── extract_subset.py      # 数据集子集提取脚本
│   ├── split_modality.py      # 模态分离与数据集划分脚本
│   ├── train.py               # 模型训练脚本
│   ├── test_model.py          # 模型测试脚本
│   ├── test_project.py        # 项目测试套件
│   ├── download_bert.py       # BERT模型下载脚本
│   └── download_qwen.py       # Qwen模型下载脚本
│
├── src/                       # 核心源码
│   ├── data/
│   │   └── data_loader.py     # 数据加载模块
│   └── model_architectures/   # 模型架构
│       ├── bert_encoder.py    # BERT文本编码器
│       ├── numeric_encoder.py # 数值特征编码器
│       ├── fusion_projection.py # 特征融合投影层
│       └── multi_modal_model.py # 多模态融合模型
│
├── utils/
│   └── log_utils.py           # 日志记录工具
│
├── docs/                      # 文档
│   └── theory.md              # 项目原理说明
│
├── data_processing/           # 原始数据集（需手动放入）
├── processed_dataset/         # 处理后数据（自动生成）
├── split_data/                # 划分后数据（自动生成）
├── saved_models/              # 训练模型（自动生成）
├── logs/                      # 操作日志（自动生成）
└── test_reports/              # 测试报告（自动生成）
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers datasets scikit-learn pandas numpy pytest matplotlib
```

### 2. 下载预训练模型

```bash
python scripts/download_bert.py
python scripts/download_qwen.py
```

### 3. 准备数据集

将原始数据集（CSV格式）放入 `data_processing/` 目录，默认数据集为 `IoT Network Intrusion Dataset.csv`，数据集需包含：
- `Label` 列：值为 `Normal`（正常流量）或 `Anomaly`（恶意流量）
- 9个连续特征列：`Flow_Duration`, `Tot_Fwd_Pkts`, `Tot_Bwd_Pkts`, `TotLen_Fwd_Pkts`, `TotLen_Bwd_Pkts`, `Flow_Byts/s`, `Fwd_Pkt_Len_Mean`, `Bwd_Pkt_Len_Mean`, `Pkt_Len_Mean`

### 4. 运行项目

打开 `main.py`，按顺序取消注释执行各步骤：

```python
# 步骤1：数据清洗（可指定数据集文件名）
from scripts.data_cleaning import main as run_data_cleaning
run_data_cleaning(dataset_filename="IoT Network Intrusion Dataset.csv")

# 步骤2：提取子集
success, dataset_id = extract_subset(num_samples=5000, random_state=42)

# 步骤3：模态分离
dataset_id = 0
split_modality(dataset_id=dataset_id, test_size=0.2, random_state=42)

# 步骤4：模型训练
train_model(model_path="./models/qwen2.5-1.5b", dataset_id=0, split_id=0, num_train_epochs=3)

# 步骤5：模型测试
result = test_model(model_id=0, dataset_id=0, split_id=0, verbose=True)

# 步骤6：绘制loss曲线
plot_loss_curve(model_id=0)
```

## 运行步骤详解

### 步骤1：数据清洗与预处理

**操作**：取消 `main.py` 中步骤1的注释

```python
from scripts.data_cleaning import main as run_data_cleaning
run_data_cleaning(dataset_filename="IoT Network Intrusion Dataset.csv")
```

**运行**：`python main.py`

**参数说明**：
- `dataset_filename`（可选）：指定要处理的CSV文件名。不指定则使用 `data_processing/` 目录下的第一个CSV文件。

**输出**：`processed_dataset/processed_dataset.csv`

**处理内容**：缺失值处理、异常值检测、标签编码（Normal→0, Anomaly→1）

**命令行直接运行**：
```bash
python scripts/data_cleaning.py --dataset "IoT Network Intrusion Dataset.csv"
```

### 步骤2：提取数据集子集

**操作**：取消 `main.py` 中步骤2的注释

```python
success, dataset_id = extract_subset(
    num_samples=5000,
    # dataset_id=0,  # 可选：指定数据集ID
    random_state=42
)
```

**运行**：`python main.py`

**参数说明**：
- `num_samples`: 提取样本数量（默认5000）
- `dataset_id`: 数据集ID（默认自动递增）
- `random_state`: 随机种子（默认42）

**输出**：
- `processed_dataset/dataset_X.csv`（子集数据）
- `processed_dataset/subset_X_scaled_features.npy`（标准化特征）
- `processed_dataset/subset_X_labels.npy`（标签）

### 步骤3：模态分离与数据集划分

**操作**：取消 `main.py` 中步骤3的注释

```python
dataset_id = 0
split_modality(
    dataset_id=dataset_id,
    # split_id=0,  # 可选：指定划分ID
    test_size=0.2,
    random_state=42
)
```

**运行**：`python main.py`

**参数说明**：
- `dataset_id`: 数据集ID（默认0）
- `split_id`: 划分ID（默认自动递增）
- `test_size`: 测试集比例（默认0.2）

**输出**：`split_data/dataset_X/split_Y/`
- `train.npz`（训练集：统计特征 + BERT嵌入 + 标签）
- `test.npz`（测试集：统计特征 + BERT嵌入 + 标签）

### 步骤4：模型训练

**操作**：取消 `main.py` 中步骤4的注释

```python
train_model(
    model_path="./models/qwen2.5-1.5b",
    # model_id=0,  # 可选：指定模型ID
    dataset_id=0,
    split_id=0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3
)
```

**运行**：`python main.py`

**参数说明**：
- `model_path`: LLM模型路径（默认`./models/qwen2.5-1.5b`）
- `model_id`: 模型ID（默认自动递增）
- `learning_rate`: 学习率（默认1e-4）
- `num_train_epochs`: 训练轮数（默认3）

**输出**：`saved_models/model_X/`
- `pytorch_model.bin`（模型参数）
- `config.txt`（训练配置）
- `loss_logs/loss_log.csv`（训练loss记录）

### 步骤5：模型测试

**操作**：取消 `main.py` 中步骤5的注释

```python
result = test_model(
    model_id=0,
    dataset_id=0,
    split_id=0,
    verbose=True
)
```

**运行**：`python main.py`

**输出**：
- 控制台打印评估指标（准确率、精确率、召回率、F1值）
- `test_reports/report_X.json`（测试报告）

### 步骤6：绘制loss曲线

**操作**：取消 `main.py` 中步骤6的注释

```python
plot_loss_curve(model_id=0)
```

**运行**：`python main.py`

**效果**：弹出窗口显示训练loss变化曲线

## GPU检查机制

所有需要GPU的程序（`split_modality.py`、`train.py`、`test_model.py`）在执行前会自动检查GPU可用性：

- **检测到GPU**：打印GPU型号和显存信息，程序正常继续
- **未检测到GPU**：发出警告，要求用户确认是否继续使用CPU运行。若选择非y则终止程序

## 主键体系

项目采用三级主键体系管理数据和模型：

| 主键 | 作用 | 示例 |
|------|------|------|
| `dataset_id` | 标识数据集子集 | `dataset_0.csv`, `dataset_1.csv` |
| `split_id` | 标识同一数据集的不同划分 | `split_data/dataset_0/split_0/` |
| `model_id` | 标识不同的训练结果 | `saved_models/model_0/` |

## 注意事项

### 硬件要求

- **GPU**：推荐 NVIDIA GPU（Ampere架构及以上），支持bfloat16混合精度训练
- **显存**：至少16GB（训练Qwen2.5-1.5B模型）
- **内存**：至少16GB（处理大数据集）

### 数据格式

- 原始数据集：CSV格式，包含`Label`列（值为`Normal`或`Anomaly`）
- 支持的特征列：`Flow_Duration`, `Tot_Fwd_Pkts`, `Tot_Bwd_Pkts`, `TotLen_Fwd_Pkts`, `TotLen_Bwd_Pkts`, `Flow_Byts/s`, `Fwd_Pkt_Len_Mean`, `Bwd_Pkt_Len_Mean`, `Pkt_Len_Mean`

### 日志系统

所有操作都会自动生成日志文件：

| 日志类型 | 保存位置 | 内容 |
|---------|---------|------|
| 子集提取 | `logs/subset/log_X.json` | 时间、数据集ID、样本数 |
| 数据划分 | `logs/split/log_X.json` | 时间、数据集ID、划分ID |
| 模型训练 | `logs/training/log_X.json` | 时间、模型ID、训练参数 |
| 模型测试 | `test_reports/report_X.json` | 时间、模型ID、评估指标 |

### 测试脚本

运行项目测试套件验证所有功能：

```bash
python -m pytest scripts/test_project.py -v
```

## 项目原理

详细的项目原理说明请参考 `docs/theory.md`。

## 数据集迁移说明

从原始Friday-WorkingHours-Afternoon-DDos数据集迁移到IoT Network Intrusion Dataset的详细改动说明，请参考 `docs/IoT_Dataset_Migration.md`。