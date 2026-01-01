import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2


def get_colormap(num_classes=256, normalized=False):
    """Generate a colormap for visualization"""
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((num_classes, 3), dtype=dtype)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class RZDGSegmentation(data.Dataset):
    """Custom RZDG Real Segmentation Dataset
    
    Args:
        root (string): Root directory of the RZDG dataset (RZDG_real_seg folder)
        image_set (string): Select the image_set to use, ``train``, ``val`` or ``test``
        transform (callable, optional): A function/transform that takes in
            PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        
    Example:
        >>> dataset = RZDGSegmentation(root='/path/to/RZDG_real_seg', image_set='train')
        >>> img, mask = dataset[0]
    """
    
    colormap = get_colormap()
    
    def __init__(self, root, image_set='RZDG_real_seg', transform=None):
        """
        Initialize the dataset
        
        Args:
            root: Path to RZDG_real_seg directory
            image_set: 'train', 'val', or 'test'
            transform: Optional transforms to be applied on both images and masks
        """
        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform
        
        # Define image and mask directories
        self.img_dir = os.path.join(self.root, 'img_dir', image_set)
        self.ann_dir = os.path.join(self.root, 'ann_dir', image_set)
        
        # Validate directories exist
        if not os.path.isdir(self.img_dir):
            raise RuntimeError(f'Image directory not found: {self.img_dir}')
        
        if image_set != 'test' and not os.path.isdir(self.ann_dir):
            raise RuntimeError(f'Annotation directory not found: {self.ann_dir}')
        
        # Get list of image files
        image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        
        if len(image_files) == 0:
            raise RuntimeError(f'No PNG images found in {self.img_dir}')
        
        # Create matching pairs
        self.images = []
        self.masks = []
        
        for img_file in image_files:
            img_path = os.path.join(self.img_dir, img_file)
            
            if image_set == 'test':
                # Test set might not have annotations
                self.images.append(img_path)
                self.masks.append(None)
            else:
                # Train/val sets must have matching annotations
                mask_path = os.path.join(self.ann_dir, img_file)
                
                if os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append(mask_path)
        
        if len(self.images) == 0:
            raise RuntimeError(f'No valid image-mask pairs found in {self.root}')
        
        print(f"Loaded {len(self.images)} images from {image_set} set")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            
        Returns:
            tuple: (image, target) where target is the segmentation mask (or None for test)
        """
        # Load image
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        
        # Load mask if available
        mask_path = self.masks[index]
        if mask_path is not None:
            target = Image.open(mask_path)
        else:
            target = None
        
        # Apply transforms
        if self.transform is not None:
            if target is not None:
                img, target = self.transform(img, target)
            else:
                img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """Decode segmentation mask to RGB image for visualization
        
        Args:
            mask (np.ndarray): Segmentation mask with shape (H, W)
            
        Returns:
            np.ndarray: RGB image with shape (H, W, 3)
        """
        return cls.colormap[mask]

    def get_dataset_info(self):
        """Print dataset information"""
        print(f"\n{'='*60}")
        print(f"Dataset: RZDG Real Segmentation")
        print(f"Image Set: {self.image_set}")
        print(f"Root: {self.root}")
        print(f"Number of samples: {len(self)}")
        print(f"Image directory: {self.img_dir}")
        if self.image_set != 'test':
            print(f"Annotation directory: {self.ann_dir}")
        print(f"{'='*60}\n")