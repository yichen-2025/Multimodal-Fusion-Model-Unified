import torch
import torch.nn as nn
import time
from transformers import AutoModel, AutoTokenizer


class BertEncoder(nn.Module):
    """
    BERT文本编码器模块
    功能：将网络流量的文本描述（如协议类型、URL路径等）编码为语义特征向量
    
    工作流程：
    1. 使用BERT预训练模型对文本进行token化
    2. 将文本输入BERT模型，提取[CLS]标记的输出作为文本语义特征
    3. 冻结BERT参数，仅作为特征提取器使用，不参与训练更新
    4. 自动检测GPU并将模型分配到GPU
    5. 支持批量推理，避免内存暴增
    """

    def __init__(self, bert_model_name="bert-base-chinese", local_model_path=None, device=None):
        """
        初始化BERT编码器
        
        Args:
            bert_model_name (str): BERT预训练模型名称，默认使用bert-base-chinese
            local_model_path (str): 本地模型目录路径，若提供则从本地加载模型（离线模式）
            device (str or torch.device): 指定设备，若为None则自动检测GPU
        """
        super().__init__()
        
        model_path = local_model_path if local_model_path is not None else bert_model_name
        
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=(local_model_path is not None))
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=(local_model_path is not None))
        
        # 自动检测并移动到GPU
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.bert = self.bert.to(self.device)
        self.hidden_size = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, text_descriptions, batch_size=32, show_progress=True):
        """
        前向传播：将文本描述转换为BERT语义特征（支持批量处理）
        
        Args:
            text_descriptions (str or list): 单个文本字符串或文本列表
            batch_size (int): 批处理大小，默认32。值越小内存占用越低，速度越慢
            show_progress (bool): 是否显示处理进度
            
        Returns:
            torch.Tensor: [CLS]标记的嵌入向量，形状为 [num_samples, hidden_size]
        """
        # 处理单个文本输入的情况，转换为列表形式
        if isinstance(text_descriptions, str):
            text_descriptions = [text_descriptions]

        total = len(text_descriptions)
        
        # 单条或少量数据直接处理，不分批
        if total <= batch_size:
            inputs = self.tokenizer(
                text_descriptions,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.bert(**inputs)
            return outputs.last_hidden_state[:, 0, :]

        # 批量处理：分批推理，最后拼接
        all_embeddings = []
        num_batches = (total + batch_size - 1) // batch_size
        start_time = time.time()
        
        for i in range(0, total, batch_size):
            batch_texts = text_descriptions[i:i + batch_size]
            batch_idx = i // batch_size + 1
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.bert(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embedding)
            
            # 进度日志
            if show_progress and (batch_idx % max(1, num_batches // 10) == 0 or batch_idx == num_batches):
                elapsed = time.time() - start_time
                pct = batch_idx / num_batches * 100
                print(f"    [BERT编码] 进度 {batch_idx}/{num_batches} ({pct:.0f}%) 耗时 {elapsed:.1f}s")

        result = torch.cat(all_embeddings, dim=0)
        
        if show_progress:
            elapsed = time.time() - start_time
            print(f"    [BERT编码] 完成: {total}样本, 耗时{elapsed:.1f}s, 设备:{self.device}")
        
        return result

    def get_tokenizer(self):
        """返回tokenizer实例"""
        return self.tokenizer

    def get_hidden_size(self):
        """返回BERT模型的隐藏层维度"""
        return self.hidden_size
