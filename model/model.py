import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from timeit import default_timer as timer

from helpers import (
    get_device, create_dataset_folders, split_data,
    get_transforms, load_data, accuracy_fn
)

#cnn model

class VehicleCNN(nn.Module):
    def __init__(self, input_feature, output_feature, hidden_unit):
        super().__init__()
        self.convo_block1 = nn.Sequential(
            nn.Conv2d(input_feature, hidden_unit, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.convo_block2 = nn.Sequential(
            nn.Conv2d(hidden_unit, hidden_unit, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_unit * 32 * 32, output_feature)
        )

    def forward(self, x):
        x = self.convo_block1(x)
        x = self.convo_block2(x)
        x = self.classifier(x)
        return x


#preparing data
def main():
    classes = ["Bus", "Car", "Truck", "motorcycle"]
    device = get_device()


    train_dir, test_dir = create_dataset_folders("/vehicle_classifier/vehicle_dataset", classes)


    split_data("/vehicle_classifier/vehicle_dataset/Dataset", train_dir, test_dir)


    train_tf, test_tf = get_transforms()


    train_data, test_data, train_loader, test_loader = load_data(
        train_dir, test_dir, train_tf, test_tf
    )


    model = VehicleCNN(3, len(classes), 10).to(device)


#training code
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    epochs = 20
    start = timer()

    for epoch in tqdm(range(epochs)):
        # TRAIN
        model.train()
        train_loss, train_acc = 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (y_pred.argmax(dim=1) == y).float().mean().item()

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc / len(train_loader))


#testing code
        model.eval()
        test_loss, test_acc = 0, 0

        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)

                test_loss += loss_fn(pred, y).item()
                test_acc += (pred.argmax(dim=1) == y).float().mean().item()

        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc / len(test_loader))

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f} | "
            f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}"
        )

    print(f"\nTraining finished in {timer() - start:.2f}s")

#saving model
    torch.save(model.state_dict(), "vehicle_cnn.pth")


if __name__ == "__main__":
    main()

