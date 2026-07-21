#Evaluate the model if detecting correctly - real and fake images
import torch
from torchvision import transforms
from PIL import Image
import os
from siamese_model import Net


def calculate_distance(img_path1, img_path2, model, transform):
    # Load and preprocess image 1
    img1 = Image.open(img_path1).convert('RGB')
    img1_tensor = transform(img1).unsqueeze(0)

    # Load and preprocess image 2
    img2 = Image.open(img_path2).convert('RGB')
    img2_tensor = transform(img2).unsqueeze(0)

    # Set model to evaluation mode (turns off dropout/batchnorm updates)
    model.eval()

    with torch.no_grad():
        # Pass both images through the trained model weights
        output1, output2 = model(img1_tensor, img2_tensor)
        # Calculate Euclidean distance between the 128-D vector coordinates
        distance = torch.nn.functional.pairwise_distance(output1, output2)

    return distance.item()


if __name__ == "__main__":
    # 1. Re-initialize the architecture layout
    model = Net()

    # 2. Load your saved weights
    model.load_state_dict(torch.load("siamese_currency_model.pth", map_location=torch.device('cpu')))
    print("🎯 Trained weights loaded successfully!")

    # 3. Define the exact same preprocessing transforms as training
    transform = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ⚠️ EDIT THESE PATHS: Pick actual images from your aligned notes folder
    IMAGE_FOLDER = "/Users/Divisha/Desktop/Currency_Detector/Data_Image/Aligned"

    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) >= 2:
        img_a = os.path.join(IMAGE_FOLDER, images[0])
        img_b = os.path.join(IMAGE_FOLDER, images[1])

        # Test 1: Compare an image with itself (Should be perfectly matching -> close to 0)
        dist_match = calculate_distance(img_a, img_a, model, transform)
        print(f"\n✅ MATCH TEST (Same Image):")
        print(f"   Calculated Distance: {dist_match:.4f}")

        # Test 2: Compare two different images
        dist_mismatch = calculate_distance(img_a, img_b, model, transform)
        print(f"\n❌ MISMATCH TEST (Different Images):")
        print(f"   Calculated Distance: {dist_mismatch:.4f}")
    else:
        print("Please check your folder path; couldn't find enough images to run the test script.")