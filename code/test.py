

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import argparse
import torchvision.transforms as transforms
from model.DRA_NET import DRA_net

# Define metrics (same as in train.py)
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
        return image, label, self.image_paths[idx]

    def __len__(self):
        return len(self.image_paths)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="DRA_net",
                      help='model name', choices=["DRA_net"])
    parser.add_argument('--model_path', required=True,
                      help='path to trained model weights')
    parser.add_argument('--img_size', type=int, default=256,
                      help='image size for resizing')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='batch size for testing')
    parser.add_argument('--output_dir', default='test_results',
                      help='directory to save test results')
    config = parser.parse_args()
    return config

def test(test_loader, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)
    
    avg_meters = {
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'f1-score': AverageMeter()
    }
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target, img_names in test_loader:
            input = input.float().cuda()
            target = target.long().cuda()
            
            # Verify target values
            if target.max() >= 3 or target.min() < 0:
                target = torch.clamp(target, 0, 2)
            
            output = model(input)
            
            # Calculate metrics
            iou = iou_score(output, target)
            dice = dice_score(output, target)
            f1 = f1_scorex(output, target)
            
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1-score'].update(f1, input.size(0))
            
            # Save predictions
            preds = torch.argmax(output, dim=1).cpu().numpy()
            for i in range(len(img_names)):
                img_name = os.path.splitext(img_names[i])[0]
                np.save(os.path.join(output_dir, 'predictions', f'{img_name}.npy'), preds[i])
                
                # Create overlay visualization
                input_img = (input[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                pred_img = np.zeros((*preds[i].shape, 3), dtype=np.uint8)
                
                # Color mapping for predictions
                pred_img[preds[i] == 1] = [255, 0, 0]  # Class 1 = Red
                pred_img[preds[i] == 2] = [0, 255, 0]  # Class 2 = Green
                
                # Create overlay (50% opacity)
                overlay = (input_img * 0.5 + pred_img * 0.5).astype(np.uint8)
                Image.fromarray(overlay).save(
                    os.path.join(output_dir, 'overlays', f'{img_name}_overlay.png'))
            
            pbar.set_postfix(OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1-score', avg_meters['f1-score'].avg)]))
            pbar.update(1)
        pbar.close()
    
    return OrderedDict([
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('f1-score', avg_meters['f1-score'].avg)])

def main():
    config = parse_args()
    
    # Load model
    model = DRA_net(3, 3).cuda()  # 3 input channels, 3 output classes
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    
    # Create dataset and loader
    test_dataset = MyData(
        '/media/iml/cv-lab/Datasets_B_cells/BCCD/test/image',
        '/media/iml/cv-lab/Datasets_B_cells/BCCD/test/mask',
        img_size=config.img_size
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Run testing
    test_log = test(test_loader, model, config.output_dir)
    
    print('\nTest Results:')
    print(f"IoU: {test_log['iou']:.4f}")
    print(f"Dice: {test_log['dice']:.4f}")
    print(f"F1-Score: {test_log['f1-score']:.4f}")

if __name__ == '__main__':
    main()