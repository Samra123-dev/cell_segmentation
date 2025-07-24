import random
import os
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Import your model
from model.DRA_NET import DRA_net

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def iou_score(output, target):
    smooth = 1e-5
    output = torch.argmax(output, dim=1)
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    intersection = (output & target).sum()
    union = (output | target).sum()
    
    return (intersection + smooth) / (union + smooth)

def dice_score(output, target):
    smooth = 1e-5
    output = torch.argmax(output, dim=1)
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    intersection = (output * target).sum()
    
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

def f1_scorex(output, target):
    smooth = 1e-5
    output = torch.argmax(output, dim=1)
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    tp = (output * target).sum()
    fp = (output * (1 - target)).sum()
    fn = ((1 - output) * target).sum()
    
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    
    return 2 * (precision * recall) / (precision + recall + smooth)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--name', default="DRA_net",
                      help='model name', choices=["DRA_net"])
    parser.add_argument('--epochs', default=50, type=int,
                      help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                      help='mini-batch size (default: 4)')
    parser.add_argument('--early_stopping', default=50, type=int,
                      help='early stopping (default: 50)')
    parser.add_argument('--num_workers', default=4, type=int)
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                      choices=['Adam', 'SGD'],
                      help='optimizer choice')
    parser.add_argument('--lr', default=0.0001, type=float,
                      help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                      help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                      help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                      help='nesterov')
    # data
    parser.add_argument('--augmentation', default=False, type=str2bool)
    parser.add_argument('--img_size', type=int, default=256,
                      help='image size for resizing')
    config = parser.parse_args()
    return config

class MyData(Dataset):
    def __init__(self, root_dir, label_dir, img_size=256):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.image_paths = sorted(os.listdir(self.root_dir))
        self.label_paths = sorted(os.listdir(self.label_dir))
        
        # Verify matching pairs
        for img, lbl in zip(self.image_paths, self.label_paths):
            assert os.path.splitext(img)[0] == os.path.splitext(lbl)[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        label_path = os.path.join(self.label_dir, self.label_paths[idx])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        image = transforms.ToTensor()(image)
        
        # Load and process label
        label = Image.open(label_path)
        if label.mode == 'RGB':
            label = label.convert('L')  # Convert to grayscale if RGB
        label = label.resize((self.img_size, self.img_size), Image.NEAREST)
        label = np.array(label)
        
        # Normalize label values to 0, 1, 2
        unique_vals = np.unique(label)
        if set(unique_vals) == {0, 255}:  # Binary mask case
            label = (label / 255).astype(np.uint8)
        elif not set(unique_vals).issubset({0,1,2}):  # Scale to 0-2 range
            label = (label / 255 * 2).round().astype(np.uint8)
        
        label = torch.from_numpy(label).long()
        return image, label

    def __len__(self):
        return len(self.image_paths)

def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                 'iou': AverageMeter(),
                 'dice': AverageMeter(),
                 'f1-score': AverageMeter()}
    
    model.train()
    pbar = tqdm(total=len(train_loader))
    
    for input, target in train_loader:
        input = input.float().cuda()
        target = target.long().cuda()
        
        # Verify target values are valid (0, 1, or 2)
        unique_vals = torch.unique(target)
        if len(unique_vals) > 3 or unique_vals.max() >= 3:
            print(f"Warning: Clipping invalid target values {unique_vals}")
            target = torch.clamp(target, 0, 2)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        
        # Metrics
        iou = iou_score(output, target)
        dice = dice_score(output, target)
        f1 = f1_scorex(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['f1-score'].update(f1, input.size(0))
        
        pbar.set_postfix(OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('f1-score', avg_meters['f1-score'].avg)]))
        pbar.update(1)
    
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                       ('iou', avg_meters['iou'].avg),
                       ('dice', avg_meters['dice'].avg),
                       ('f1-score', avg_meters['f1-score'].avg)])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                 'iou': AverageMeter(),
                 'dice': AverageMeter(),
                 'f1-score': AverageMeter()}
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.float().cuda()
            target = target.long().cuda()
            
            # Verify target values
            if target.max() >= 3 or target.min() < 0:
                target = torch.clamp(target, 0, 2)
            
            output = model(input)
            loss = criterion(output, target)
            
            iou = iou_score(output, target)
            dice = dice_score(output, target)
            f1 = f1_scorex(output, target)
            
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1-score'].update(f1, input.size(0))
            
            pbar.set_postfix(OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1-score', avg_meters['f1-score'].avg)]))
            pbar.update(1)
        pbar.close()
    
    return OrderedDict([('loss', avg_meters['loss'].avg),
                       ('iou', avg_meters['iou'].avg),
                       ('dice', avg_meters['dice'].avg),
                       ('f1-score', avg_meters['f1-score'].avg)])

def main():
    # Initialize
    config = parse_args()
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    
    # Create output directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Model
    model = DRA_net(3, 3).cuda()  # 3 input channels, 3 output classes
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, 
                            momentum=config.momentum, 
                            nesterov=config.nesterov,
                            weight_decay=config.weight_decay)
    
    # Data
    train_dataset = MyData(
        'give your training image directory path',
        'give your training masks directory path',
        img_size=config.img_size
    )
    val_dataset = MyData(
        'give your validation image directory path',
        'give your validation image directory path',
        img_size=config.img_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                          shuffle=False, num_workers=config.num_workers)
    
    # Training
    best_dice = 0
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")
        train_log = train(train_loader, model, criterion, optimizer)
        val_log = validate(val_loader, model, criterion)
        
        # Save best model
        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), f'checkpoints/best_model_{config.name}.pth')
            best_dice = val_log['dice']
            print(f"Saved new best model with dice: {best_dice:.4f}")
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()