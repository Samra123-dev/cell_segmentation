import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from model.DRA_NET import DRA_net
import matplotlib.pyplot as plt

class InferenceData(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_paths = sorted(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        image_tensor = transforms.ToTensor()(image)
        
        return image_tensor, img_name

    def __len__(self):
        return len(self.image_paths)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to trained model weights')
    parser.add_argument('--img_dir', required=True, help='directory with input images')
    parser.add_argument('--output_dir', default='inference_results', help='where to save predictions')
    parser.add_argument('--img_size', type=int, default=256, help='resize image size')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for inference')
    return parser.parse_args()

def inference(loader, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(loader))
        for inputs, img_names in loader:
            inputs = inputs.float().cuda()

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for i in range(len(img_names)):
                img_name = os.path.splitext(img_names[i])[0]
                
                # Save raw prediction
                np.save(os.path.join(output_dir, 'predictions', f'{img_name}.npy'), preds[i])

                # Create overlay
                input_img = (inputs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                pred_img = np.zeros((*preds[i].shape, 3), dtype=np.uint8)
                pred_img[preds[i] == 1] = [255, 0, 0]
                pred_img[preds[i] == 2] = [0, 255, 0]
                overlay = (input_img * 0.5 + pred_img * 0.5).astype(np.uint8)

                # Save overlay
                Image.fromarray(overlay).save(
                    os.path.join(output_dir, 'overlays', f'{img_name}_overlay.png'))

                # Display side-by-side
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(input_img)
                plt.title(f"Original: {img_name}")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(overlay)
                plt.title("Predicted Overlay")
                plt.axis('off')

                plt.tight_layout()
                plt.show()

            pbar.update(1)
        pbar.close()

def main():
    config = parse_args()
    
    model = DRA_net(3, 3).cuda()
    model.load_state_dict(torch.load(config.model_path))
    
    dataset = InferenceData(
        root_dir=config.img_dir,
        img_size=config.img_size
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )

    inference(loader, model, config.output_dir)

if __name__ == '__main__':
    main()



