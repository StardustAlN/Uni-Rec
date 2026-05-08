#!/usr/bin/env python3
"""OneTrans 基础版训练脚本"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入现有代码
sys.path.insert(0, str(Path(__file__).parent))
from dataset import PCVRParquetDataset, get_pcvr_data
from onetrans_model import OneTransModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_schema(schema_path: str) -> dict:
    """加载 schema 配置"""
    with open(schema_path, 'r') as f:
        return json.load(f)


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_preds = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # 移动数据到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # 前向传播
        preds = model(batch)
        
        # 计算损失
        labels = batch['label'].float()
        loss = criterion(preds, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        total_preds += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_preds


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_preds = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        preds = model(batch)
        labels = batch['label'].float()
        loss = criterion(preds, labels)
        
        total_loss += loss.item() * labels.size(0)
        total_preds += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_preds
    
    # 计算 AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5
    
    return avg_loss, auc


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 使用现有的数据加载函数（不使用 valid_ratio，全部作为训练集）
    train_loader, val_loader, train_dataset = get_pcvr_data(
        data_dir=os.path.dirname(args.train_data),
        schema_path=args.schema_path,
        batch_size=args.batch_size,
        valid_ratio=0.0,  # 不使用验证集分割
        num_workers=args.num_workers,
        shuffle_train=True
    )
    
    logger.info(f"Train/Val loaders created")
    
    # 获取 schema (直接从 train_dataset 获取属性)
    schema_info = {
        'user_int_dim': train_dataset.user_int_schema.total_dim,
        'item_int_dim': train_dataset.item_int_schema.total_dim,
        'user_dense_dim': train_dataset.user_dense_schema.total_dim,
        'seq_fields': train_dataset.seq_domains
    }
    logger.info(f"Schema info: {schema_info}")
    
    # 创建模型配置
    config = {
        'user_int_dim': schema_info['user_int_dim'],
        'item_int_dim': schema_info['item_int_dim'],
        'user_dense_dim': schema_info['user_dense_dim'],
        'item_dense_dim': 0,  # item_dense_feats 为空
        'seq_fields': schema_info['seq_fields'],
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'n_layers': args.n_layers,
        'max_seq_len': args.max_seq_len
    }
    
    logger.info(f"Model config: {config}")
    
    # 创建模型
    model = OneTransModel(config).to(device)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}\nEpoch {epoch}/{args.epochs}\n{'='*50}")
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # 学习率调度
        scheduler.step()
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': val_auc,
                'config': config
            }, args.output_dir / 'best_onetrans.pth')
            logger.info(f"Saved best model with AUC: {val_auc:.4f}")
    
    logger.info(f"\nTraining completed! Best Val AUC: {best_auc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OneTrans Model')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--val_data', type=str, required=True, help='验证数据路径')
    parser.add_argument('--schema_path', type=str, required=True, help='Schema 文件路径')
    parser.add_argument('--output_dir', type=Path, default=Path('./outputs'), help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=512, help='FFN 维度')
    parser.add_argument('--n_layers', type=int, default=4, help='Transformer 层数')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)
