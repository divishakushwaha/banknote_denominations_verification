import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SiameseIndianCurrencyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # Scan for images
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(self.image_files) < 2:
            raise ValueError(
                f"CRITICAL: You need at least 2 raw images inside '{image_dir}' to construct relative pairs.")

        # Uniform structural preparation rules applied prior to feature scaling
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 448)),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.1)),
                transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        # We synthesize structural data variances dynamically by upscaling the epoch duration footprint
        return len(self.image_files) * 4

    def __getitem__(self, idx):
        # Probabilistic allocation flag decides whether to yield a targeted or divergent cluster pair
        # 1.0 = Target Match (Pull vectors close), 0.0 = Distant Discrepancy (Push vectors apart)
        should_match = random.choice([1.0, 0.0])

        # Primary Anchor sample instantiation
        img1_name = self.image_files[idx % len(self.image_files)]
        img1_path = os.path.join(self.image_dir, img1_name)
        img1 = Image.open(img1_path).convert('RGB')

        if should_match == 1.0:
            # POSITIVE MATCH: Anchor note is paired directly with its matching geometric instance
            img2_path = img1_path
            label = torch.tensor([1.0], dtype=torch.float32)
        else:
            # NEGATIVE MISMATCH: Anchor note is paired with a variant note out of distribution
            img2_name = random.choice(self.image_files)
            while img2_name == img1_name:
                img2_name = random.choice(self.image_files)

            img2_path = os.path.join(self.image_dir, img2_name)
            label = torch.tensor([0.0], dtype=torch.float32)

        img2 = Image.open(img2_path).convert('RGB')

        # Formatting implementation step across color spatial layers
        if self.transform:
            img1_tensor = self.transform(img1)
            img2_tensor = self.transform(img2)

        return img1_tensor, img2_tensor, label


# =====================================================================
# LOCAL ARCHITECTURAL SMOKE TEST
# =====================================================================
if __name__ == "__main__":
    IMAGE_DIRECTORY = "/Users/Divisha/Desktop/Currency_Detector/Data_Image/Aligned"

    try:
        dataset = SiameseIndianCurrencyDataset(image_dir=IMAGE_DIRECTORY)
        print(f"📦 Pipeline Status: Successfully mapped {len(dataset)} dynamic spatial pairs.")

        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        img1_batch, img2_batch, label_batch = next(iter(dataloader))

        print("\n=== Validation Verification Log ===")
        print(f"• Target Anchor Tensor Matrix Dimensions : {img1_batch.shape}")
        print(f"• Comparison Vector Matrix Dimensions   : {img2_batch.shape}")
        print(f"• Dynamic Binary Target Allocations     : {label_batch.squeeze().tolist()}")
        print("💡 Status Check: Pipeline functional. Structural parameters ready for execution training loop.")

    except Exception as e:
        print(f"❌ Pipeline Initialization Error: {e}")