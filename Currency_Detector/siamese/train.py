import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SiameseIndianCurrencyDataset
from siamese_model import Net


# 1. Define the Custom Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):  # Dropping margin to 1.0 makes it easier to stabilize
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2, keepdim=True)

        # label=1 means MATCHING images (Distance should be 0)
        # label=0 means MISMATCHING images (Distance should be pushed past the margin)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


# 2. Set Up the Training Configurations
def train():
    # Automatically use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Initialize Dataset and DataLoader
    dataset = SiameseIndianCurrencyDataset(image_dir="/Users/Divisha/Desktop/Currency_Detector/Data_Image/Aligned")  # Put your real note folder path here
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the Model and move it to the GPU/CPU
    model = Net().to(device)

    # Initialize our custom loss and optimizer (Adam handles learning rates smoothly)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 3. The Core Training Loop
    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (img1, img2, label) in enumerate(dataloader):
            # Move data tensors to the processing device
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # Clear out old gradients from the previous step
            optimizer.zero_grad()

            # Forward Pass: Send both images through the twin network branches
            out1, out2 = model(img1, img2)

            # Calculate how far off the distances are from the target labels
            loss = criterion(out1, out2, label)

            # Backward Pass: Calculate the gradients (the error rates)
            loss.backward()

            # Optimization Step: Tweak the network filters to perform better next time
            optimizer.step()

            running_loss += loss.item()

        print(f"📊 Epoch [{epoch + 1}/{epochs}] complete. Average Loss: {running_loss / len(dataloader):.4f}")

    # Save the trained model weights so we can use it to verify notes later!
    torch.save(model.state_dict(), "siamese_currency_model.pth")
    print("🎉 Training finished and model saved successfully!")


if __name__ == "__main__":
    train()