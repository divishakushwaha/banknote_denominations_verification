import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SiameseIndianCurrencyDataset
from siamese_model import Net

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):  
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training on device: {device}")
    dataset = SiameseIndianCurrencyDataset(image_dir="/Users/Divisha/Desktop/Currency_Detector/Data_Image/Aligned")  # Put your real note folder path here
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = Net().to(device)

    # Adam Optimization
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #Training 
    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (img1, img2, label) in enumerate(dataloader):
          
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"📊 Epoch [{epoch + 1}/{epochs}] complete. Average Loss: {running_loss / len(dataloader):.4f}")

    # Saved the trained model weights 
    torch.save(model.state_dict(), "siamese_currency_model.pth")
    print(" Training finished and model saved successfully!")


if __name__ == "__main__":
    train()
