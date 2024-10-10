import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image

# Import the helper functions
import transforms as T
import utils
from engine import train_one_epoch, evaluate

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Import tqdm for the progress bar
from tqdm import tqdm


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        # Get all image files, exclude mask files
        self.imgs = [f for f in os.listdir(self.root)
                     if f.endswith('.JPG') and not f.endswith('_mask.JPG')]
        self.imgs.sort()

        print(f"Found {len(self.imgs)} images in {self.root}")

        # Filter out images without masks
        self.valid_imgs = []
        for img_name in self.imgs:
            mask_name = img_name.replace('.JPG', '_mask.npy')
            mask_path = os.path.join(self.root, mask_name)
            if not os.path.exists(mask_path):
                continue
            mask = np.load(mask_path)
            obj_ids = np.unique(mask)
            if len(obj_ids) > 1:  # At least one object exists
                self.valid_imgs.append(img_name)

        print(f"Dataset size after filtering images with no masks: {len(self.valid_imgs)}")

    def __getitem__(self, idx):
        img_name = self.valid_imgs[idx]
        img_path = os.path.join(self.root, img_name)

        # Generate the corresponding mask filename
        mask_name = img_name.replace('.JPG', '_mask.npy')
        mask_path = os.path.join(self.root, mask_name)

        # Load image and mask
        img = Image.open(img_path).convert("RGB")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file {mask_name} not found for image {img_name}")

        mask = np.load(mask_path)

        # Instances are encoded as different colors
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # Remove background (assumed to be 0)

        # Split the mask into multiple masks, one for each object
        masks = mask == obj_ids[:, None, None]

        # Get bounding boxes for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert everything into torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # All objects are of class 1
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # Assume all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.valid_imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256

    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=hidden_layer,
        num_classes=num_classes)

    return model


def main():
    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2  # Background and one object class

    # Create the dataset
    dataset = CustomDataset(
        root='C:/Users/polis/Downloads/opensource_w_masks_all/train',
        transforms=get_transform(train=True),
    )
    print(f"Training dataset size: {len(dataset)}")

    dataset_test = CustomDataset(
        root='C:/Users/polis/Downloads/opensource_w_masks_all/test',
        transforms=get_transform(train=False),
    )
    print(f"Test dataset size: {len(dataset_test)}")

    # Define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # Get the model
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    # Number of epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # Train for one epoch with progress bar
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_one_epoch_with_progress(model, optimizer, data_loader, device, epoch, print_freq=10)

        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


def train_one_epoch_with_progress(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    # Use tqdm for the progress bar
    for images, targets in tqdm(data_loader, desc=header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_dict
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced.item(), **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


if __name__ == "__main__":
    main()
