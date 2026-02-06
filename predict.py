from torch.utils.data import dataset
import network
from utils import ext_transforms as et
import os
import argparse
import numpy as np
from datasets import voc
from torchvision import transforms as T
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from datasets import voc

matplotlib.use('Agg')  


class PredictConfig:
    # Model
    model_name = 'deeplabv3plus_mobilenet'
    num_classes = 3
    output_stride = 16
    checkpoint = './checkpoints/best_model_250.pth'
    
    # Data
    test_dir = './RZDG_real_seg/img_dir/test'
    output_dir = './results_250'
    
    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    
    # Image settings
    image_size = 513

    def get_colormap(num_classes=256):
        """Get colormap for visualization"""
        # Hardcoded colormap for 3 classes: Background, Barrier, Road Beacon
        # Background: Black (0, 0, 0)
        # Barrier: Red (255, 0, 0)
        # Road Beacon: Yellow (255, 255, 0)
        return np.array([
            [0, 0, 0],       # Class 0: Background
            [255, 0, 0],     # Class 1: Barrier
            [255, 255, 0],   # Class 2: Road Beacon
        ], dtype=np.uint8)


def load_model(cfg):
    """Load trained model from checkpoint"""
    print(f"Loading model: {cfg.model_name}")
    
    # Create model
    model_func = network.modeling.__dict__[cfg.model_name]
    model = model_func(
        num_classes=cfg.num_classes,
        output_stride=cfg.output_stride,
        pretrained_backbone=False
    )
    
    # Load checkpoint
    if os.path.exists(cfg.checkpoint):
        print(f"Loading checkpoint: {cfg.checkpoint}")
        checkpoint = torch.load(cfg.checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")
    
    model.to(cfg.device)
    model.eval()
    
    return model



# PREPROCESSING

def get_transform(image_size=513):
    """Get preprocessing transform"""
    return et.ExtCompose([
        et.ExtResize(size=(image_size, image_size)),
        et.ExtToTensor(normalize=True, target_type='uint8'),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
    ])


def predict_single_image(model, image_path, transform, cfg):
    """Predict segmentation for a single image"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_img = np.array(img)
    
    # Transform
    img_tensor, _ = transform(img, Image.new('L', img.size))
    img_tensor = img_tensor.unsqueeze(0).to(cfg.device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    # Resize prediction back to original size
    pred_pil = Image.fromarray(pred)
    pred_pil = pred_pil.resize(img.size, resample=Image.NEAREST)
    pred = np.array(pred_pil)

    return original_img, pred


def colorize_mask(pred, colormap):
    """Convert prediction mask to RGB image"""
    colored = colormap[pred]
    return Image.fromarray(colored)


def visualize_result(original_img, pred_mask, colormap, output_path=None):
    """Visualize original image, prediction and overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    pred_colored = colormap[pred_mask]
    axes[1].imshow(pred_colored)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Overlay
    alpha = 0.5
    overlay = (original_img * (1 - alpha) + pred_colored * alpha).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_overlay_image(original_img, pred_mask, colormap, output_path, alpha=0.5):
    """Save overlay image only"""
    pred_colored = colormap[pred_mask]
    overlay = (original_img * (1 - alpha) + pred_colored * alpha).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="path to image or directory (default: test_dir from config)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="output directory")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="number of classes")
    parser.add_argument("--masks", action='store_true', default=False,
                        help="save prediction masks")
    parser.add_argument("--visualize", action='store_true', default=False,
                        help="save 3-panel visualization")
    parser.add_argument("--overlay", action='store_true', default=False,
                        help="save overlay image only")
    
    args = parser.parse_args()
    
    # Setup config
    cfg = PredictConfig()
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint
    if args.output:
        cfg.output_dir = args.output
    if args.num_classes:
        cfg.num_classes = args.num_classes
    
    input_path = args.input if args.input else cfg.test_dir
    
    # Default behavior: if no output flags are set, save masks + visualization + overlay
    if not (args.masks or args.visualize or args.overlay):
        args.masks = True
        args.visualize = True
        args.overlay = True

    # Create output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    masks_dir = os.path.join(cfg.output_dir, 'masks')
    vis_dir = os.path.join(cfg.output_dir, 'visualization')
    overlay_dir = os.path.join(cfg.output_dir, 'overlay')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Load model
    model = load_model(cfg)
    
    # Get colormap
    colormap = PredictConfig.get_colormap(cfg.num_classes)
    
    # Get transform
    transform = get_transform(cfg.image_size)
    
    # Get image files
    image_files = []
    if os.path.isdir(input_path):
        for ext in ['png', 'jpg', 'jpeg']:
            files = glob(os.path.join(input_path, f'*.{ext}'))
            image_files.extend(files)
        image_files.extend(glob(os.path.join(input_path, f'*.PNG')))
        image_files.extend(glob(os.path.join(input_path, f'*.JPG')))
    elif os.path.isfile(input_path):
        image_files.append(input_path)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    image_files = sorted(image_files)
    print(f"\nFound {len(image_files)} images")
    print("="*60)
    
    # Process images
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing", unit="img"), 1):
        img_name = os.path.basename(img_path).split('.')[0]
        
        try:
            # Predict once
            original_img, pred_mask = predict_single_image(model, img_path, transform, cfg)

            if args.overlay:
                overlay_path = os.path.join(overlay_dir, f'{img_name}_overlay.png')
                save_overlay_image(original_img, pred_mask, colormap, overlay_path)
                # If overlay-only, skip other outputs unless explicitly requested
                if not (args.masks or args.visualize):
                    continue

            if args.masks:
                # Save prediction mask
                pred_colored = Image.fromarray(colormap[pred_mask])
                pred_path = os.path.join(masks_dir, f'{img_name}_pred.png')
                pred_colored.save(pred_path)

            # Save visualization
            if args.visualize:
                vis_path = os.path.join(vis_dir, f'{img_name}_vis.png')
                visualize_result(original_img, pred_mask, colormap, vis_path)

                
        
        except Exception as e:
            print(f"  ✗ Error processing {img_name}: {str(e)}")
    
    print("="*60)
    print(f"Results saved to: {cfg.output_dir}")


if __name__ == '__main__':
    main()
