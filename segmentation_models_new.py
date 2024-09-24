# Import necessary libraries
import os
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the function to create masks from JSON files
def create_mask_from_json(json_data, image_size):
    width, height = image_size
    mask = Image.new('L', (width, height), 0)  # 'L' mode for (8-bit pixels, black and white)
    draw = ImageDraw.Draw(mask)
    for shape in json_data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = shape['points']
            # Convert points to tuple of (x, y)
            polygon = [tuple(point) for point in points]
            draw.polygon(polygon, outline=1, fill=1)
    return np.array(mask)

# Paths to your data
data_dir = 'SensorCommunication/Acquisition/training_opensource_datasets'

# Lists to store images and masks
images = []
masks = []

# Iterate over files in the directory
for file_name in os.listdir(data_dir):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(data_dir, file_name)
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Load corresponding mask
        base_name = file_name.rsplit('.', 1)[0]
        mask_name = base_name + '.json'
        mask_path = os.path.join(data_dir, mask_name)
        if os.path.exists(mask_path):
            with open(mask_path, 'r') as f:
                mask_data = json.load(f)
                # Convert JSON annotations to mask
                mask_np = create_mask_from_json(mask_data, image.size)
                images.append(image_np)
                masks.append(mask_np)
        else:
            # If no corresponding JSON file, skip this image
            print(f"No mask found for {file_name}, skipping.")
            continue

# Convert lists to NumPy arrays
images = np.array(images)
masks = np.array(masks)

# Define transformations for images and masks
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize images to a consistent size
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

# Define your dataset class
class LeafDataset(Dataset):
    def __init__(self, images, masks, transform=None, mask_transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = torch.squeeze(mask)  # Remove channel dimension
        return img, mask.long()

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_dataset = LeafDataset(train_images, train_masks, transform=transform, mask_transform=mask_transform)
val_dataset = LeafDataset(val_images, val_masks, transform=transform, mask_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Define the segmentation model (U-Net)
class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # Conv1 and MaxPool
        self.layer0_pool = self.base_layers[3]
        self.layer1 = self.base_layers[4]
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_up4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)
        x0p = self.layer0_pool(x0)
        x1 = self.layer1(x0p)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Decoder with skip connections
        x = self.up1(x4)
        x = self.conv_up1(torch.cat([x, x3], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))
        
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x1], dim=1))
        
        x = self.up4(x)
        x = self.conv_up4(torch.cat([x, x0], dim=1))
        
        x = self.out_conv(x)
        return x

# Initialize the model, loss function, and optimizer
num_classes = 2  # Background and leaf
model = UNet(n_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Initial training with labeled data
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    
    # Validate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

# Generate pseudo labels for unlabeled data (if available)
# For this example, we'll assume there is no unlabeled data
# If you have unlabeled images, you can load them and generate pseudo labels as shown below

unlabeled_data_dir = 'SensorCommunication/Acquisition/testing_opensource_datasets'  # Update this path
unlabeled_images = []

# Load unlabeled images
for file_name in os.listdir(unlabeled_data_dir):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(unlabeled_data_dir, file_name)
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        unlabeled_images.append(image_np)

# Create a dataset for unlabeled images
unlabeled_dataset = LeafDataset(unlabeled_images, [np.zeros((256, 256))]*len(unlabeled_images),
                                transform=transform, mask_transform=mask_transform)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=False, num_workers=2)

# Generate pseudo labels using the trained model
pseudo_labels = []
model.eval()
with torch.no_grad():
    for imgs, _ in tqdm(unlabeled_loader, desc="Generating Pseudo Labels"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        pseudo_labels.extend(preds.cpu().numpy())

# Implement the mediator network for pseudo label denoising
class MediatorNet(nn.Module):
    def __init__(self):
        super(MediatorNet, self).__init__()
        # Simple CNN for denoising
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

mediator = MediatorNet().to(device)
mediator_optimizer = torch.optim.Adam(mediator.parameters(), lr=1e-4)
mediator_criterion = nn.BCELoss()

# Prepare pseudo labels and corresponding images for mediator training
# Since we don't have ground truth for pseudo labels, we'll use the original masks for simplicity
# In practice, you would have a small set of data with ground truth to train the mediator

mediator_dataset = []
for img_np, pseudo_label_np in zip(unlabeled_images, pseudo_labels):
    mediator_dataset.append((img_np, pseudo_label_np))

mediator_loader = DataLoader(mediator_dataset, batch_size=4, shuffle=True, num_workers=2)

# Training the mediator network
mediator_epochs = 10
for epoch in range(mediator_epochs):
    mediator.train()
    epoch_loss = 0
    for img_np, pseudo_label_np in mediator_loader:
        pseudo_label_tensor = pseudo_label_np.unsqueeze(1).float().to(device)
        mediator_optimizer.zero_grad()
        denoised_label = mediator(pseudo_label_tensor)
        loss = mediator_criterion(denoised_label, pseudo_label_tensor)
        loss.backward()
        mediator_optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(mediator_loader)
    print(f"Mediator Epoch {epoch+1}/{mediator_epochs}, Loss: {avg_loss:.4f}")

# Denoise the pseudo labels
denoised_pseudo_labels = []
mediator.eval()
with torch.no_grad():
    for pseudo_label_np in pseudo_labels:
        pseudo_label_tensor = torch.tensor(pseudo_label_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        denoised_label = mediator(pseudo_label_tensor)
        denoised_label_np = denoised_label.squeeze().cpu().numpy()
        denoised_pseudo_labels.append(denoised_label_np)

# Combine original labeled data with denoised pseudo labels
combined_images = np.concatenate((train_images, unlabeled_images), axis=0)
combined_masks = np.concatenate((train_masks, denoised_pseudo_labels), axis=0)

# Create new dataset and dataloader for retraining
combined_dataset = LeafDataset(combined_images, combined_masks, transform=transform, mask_transform=mask_transform)
combined_loader = DataLoader(combined_dataset, batch_size=4, shuffle=True, num_workers=2)

# Retrain the segmentation model with combined data
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for imgs, masks in tqdm(combined_loader, desc=f"Retraining Epoch {epoch+1}/{num_epochs}"):
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(combined_loader)
    print(f"Retraining Loss: {avg_loss:.4f}")

    # Validate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

# Evaluate the model on the validation set
def compute_dice_score(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

model.eval()
total_dice = 0
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        preds = preds.cpu().numpy()
        masks = masks.cpu().numpy()
        for pred, mask in zip(preds, masks):
            dice_score = compute_dice_score(pred.flatten(), mask.flatten())
            total_dice += dice_score
average_dice = total_dice / len(val_dataset)
print(f"Average Dice Score on Validation Set: {average_dice:.4f}")

# Visualize some predictions
import random

model.eval()
with torch.no_grad():
    for i in range(5):  # Show 5 random examples
        idx = random.randint(0, len(val_dataset) - 1)
        img, mask = val_dataset[idx]
        img_input = img.unsqueeze(0).to(device)
        output = model(img_input)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img.permute(1, 2, 0).numpy())
        ax[0].set_title('Input Image')
        ax[1].imshow(mask.numpy(), cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[2].imshow(pred, cmap='gray')
        ax[2].set_title('Predicted Mask')
        plt.show()
