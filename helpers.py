import os
import random
import shutil
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


#GPU OR CPU
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#creating folders
def create_dataset_folders(base_dir="vehicle_classifier/vehicle_data/Dataset", classes=None):
    if classes is None:
        classes = ["Bus", "Car", "Truck", "motorcycle"]

    os.makedirs(base_dir, exist_ok=True)

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    return train_dir, test_dir

#spliting data
def split_data(source_folder, train_dir, test_dir, split_ratio=0.8):
    for cls in os.listdir(source_folder):
        cls_path = os.path.join(source_folder, cls)
        imgs = os.listdir(cls_path)
        random.shuffle(imgs)
        split = int(split_ratio * len(imgs))

        for img in imgs[:split]:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(train_dir, cls, img))

        for img in imgs[split:]:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(test_dir, cls, img))


#transforms
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    return train_transform, test_transform


#loading data
def load_data(train_dir, test_dir, train_tf, test_tf, batch_size=32):
    train_data = datasets.ImageFolder(train_dir, transform=train_tf)
    test_data = datasets.ImageFolder(test_dir, transform=test_tf)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    return train_data, test_data, train_loader, test_loader

#accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

