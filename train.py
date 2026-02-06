import os
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import network
from datasets.voc import RZDGSegmentation
from utils import ext_transforms as et
from utils.loss import FocalLoss
from utils.scheduler import PolyLR 
from metrics.stream_metrics import StreamSegMetrics
from utils.visualizer import Visualizer


class Config:
    # Dataset

    data_root = os.getenv('DATA_ROOT', './RZDG_real_seg')
    num_classes = 3  # Change to 1 for binary segmentation
    
    # Model
    model_name = 'deeplabv3plus_mobilenet'
    output_stride = 16
    pretrained_backbone = True
    separable_conv = False
    
    # Training
    batch_size = 13
    num_workers = 2
    learning_rate = 0.001
    weight_decay = 1e-4
    momentum = 0.9
    epochs = 200
    total_itrs = 30000  # Total iterations
    
    # Augmentation
    image_size = 513
    crop_size = 513

    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Checkpoints
    load_checkpoint = True
    ckpt_dir = './checkpoints'
    results_dir = './results'
    log_dir = './runs'
    
    # Logging
    print_interval = 10
    val_interval = 100


def get_train_transforms(crop_size=513):
    """Data augmentation for training"""
    return et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(p=0.5),
        et.ExtRandomVerticalFlip(p=0.1),
        et.ExtRandomRotation(degrees=10),
        et.ExtColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        et.ExtToTensor(normalize=True, target_type='int64'),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size=513):
    """No augmentation for validation"""
    return et.ExtCompose([
        et.ExtResize(size=(image_size, image_size)),
        et.ExtToTensor(normalize=True, target_type='int64'),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
    ])


def create_dataloaders(cfg):
    """Create train and validation dataloaders"""
    
    # Training dataset
    train_dst = RZDGSegmentation(
        root=cfg.data_root,
        image_set='train',
        transform=get_train_transforms(cfg.crop_size)
    )
    
    # Validation dataset
    val_dst = RZDGSegmentation(
        root=cfg.data_root,
        image_set='val',
        transform=get_val_transforms(cfg.image_size)
    )
    
    train_loader = data.DataLoader(
        train_dst,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True
    )
    
    val_loader = data.DataLoader(
        val_dst,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    
    print(f"Train dataset: {len(train_dst)} images")
    print(f"Val dataset: {len(val_dst)} images")
    
    return train_loader, val_loader, len(train_dst), len(val_dst)


def create_model(cfg):
    """Create DeepLabV3+ with MobileNetV2 backbone"""
    
    print(f"Creating model: {cfg.model_name}")
    
    # Get model builder function
    model_func = network.modeling.__dict__[cfg.model_name]
    
    # Create model
    model = model_func(
        num_classes=cfg.num_classes,
        output_stride=cfg.output_stride,
        pretrained_backbone=cfg.pretrained_backbone
    )
    
    # Apply separable convolution if needed
    if cfg.separable_conv and cfg.model_name != 'deeplabv3plus_mobilenet':
        network.convert_to_separable_conv(model.classifier)
    
    return model


def main(cfg):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=cfg.log_dir)
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    train_loader, val_loader, train_size, val_size = create_dataloaders(cfg)
    
    model = create_model(cfg)
    model.to(cfg.device)
    
    # Load checkpoint if exists
    start_epoch = 0
    checkpoint_path = os.path.join(cfg.ckpt_dir, 'best_model.pth')
    
    if cfg.load_checkpoint:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
            model.load_state_dict(checkpoint)
            print("✓ Checkpoint loaded successfully!")
    
    criteria = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    params_to_optimize = [
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg.learning_rate},
        {'params': [p for p in model.classifier.parameters() if p.requires_grad], 'lr': cfg.learning_rate * 10},
    ]
    
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    
    scheduler = PolyLR(
        optimizer,
        max_iters=cfg.total_itrs,
        power=0.9
    )
    
    metrics = StreamSegMetrics(cfg.num_classes)
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    print(f"Device: {cfg.device}")
    print(f"Model: {cfg.model_name}")
    print(f"Starting from epoch: {start_epoch + 1}")
    print(f"Total epochs: {cfg.epochs}")
    print("="*60 + "\n")
    
    best_score = 0.0
    iteration = 0
    
    # Start from checkpoint epoch
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{cfg.epochs}', unit='batch') as pbar:
            for images, labels in train_loader:
                images = images.to(cfg.device, dtype=torch.float32)
                labels = labels.to(cfg.device, dtype=torch.long)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                
                # Loss calculation
                loss = criteria(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                iteration += 1
                
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log training loss to TensorBoard
                writer.add_scalar('Train/Loss', loss.item(), iteration)
                
                # Print loss
                # if iteration % cfg.print_interval == 0:
                #     print(f"Iteration {iteration}/{cfg.total_itrs} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            model.eval()
            metrics.reset()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc='Validation'):
                    images = images.to(cfg.device, dtype=torch.float32)
                    labels = labels.to(cfg.device, dtype=torch.long)
                    
                    outputs = model(images)
                    loss = criteria(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    # Update metrics
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    labels = labels.cpu().numpy()
                    metrics.update(labels, preds)
            
            avg_val_loss = val_loss / len(val_loader)
            results = metrics.get_results()
            
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  mIoU: {results['Mean IoU']:.4f}")
            print(f"  Accuracy: {results['Overall Acc']:.4f}\n")

            # Log validation metrics to TensorBoard
            writer.add_scalar('Val/Loss', avg_val_loss, epoch + 1)
            writer.add_scalar('Val/mIoU', results['Mean IoU'], epoch + 1)
            writer.add_scalar('Val/Accuracy', results['Overall Acc'], epoch + 1)
            
            # Save best model
            if results['Mean IoU'] > best_score:
                best_score = results['Mean IoU']
                torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'best_model.pth'))
                print(f"✓ Best model saved with mIoU: {best_score:.4f}\n")
 
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, f'model_epoch_{epoch+1}.pth'))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print(f"Best mIoU: {best_score:.4f}")
    print("="*60)
    
    writer.close()


if __name__ == '__main__':
    cfg = Config()
    main(cfg)