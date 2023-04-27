import typing as T

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import Tensor
from torchvision.transforms import Compose
import fire

from .model import UNet
from .utils import DiceLoss, RMSELoss
from .data import (
    FISHSpotsDataset, RandomHorizontalFlip,
    RandomRotation, ToTensorWrapper,
)


def train(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        criterion: T.Callable[[Tensor, Tensor], Tensor],
        writer: SummaryWriter,
        device: torch.device,
        train_loader: DataLoader, test_loader: DataLoader,
        model_save_path: str,
        num_epochs=50):
    """
    The training loop.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        criterion: The loss function to use.
        writer: The TensorBoard writer.
        device: The device to use.
        train_loader: The training data loader.
        test_loader: The test data loader.
        model_save_path: The path to save the model to.
        num_epochs: The number of epochs to train for.
    """
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for idx, batch in enumerate(train_loader):
            images = batch["image"].to(device, dtype=torch.float)
            masks = batch["mask"].to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}/{num_epochs}, "
                    f"Batch: {idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
                writer.add_scalar(
                    "Loss/train_batch", loss.item(),
                    epoch * len(train_loader) + idx)
                img = images[0, 0].cpu().numpy()
                img = np.stack((img,)*3, axis=0)
                # normalize to 0-255
                img = (img - img.min()) / (img.max() - img.min()) * 255
                # record images
                writer.add_image(
                    "Image/input", images[0], epoch * len(train_loader) + idx)
                writer.add_image(
                    "Image/mask", masks[0], epoch * len(train_loader) + idx)
                writer.add_image(
                    "Image/pred", outputs[0], epoch * len(train_loader) + idx)

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device, dtype=torch.float)
                masks = batch["mask"].to(device, dtype=torch.float)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with Val Loss: {val_loss:.4f}")

    writer.close()


def main(
        meta_train_path: str, meta_test_path: str,
        dataset_root_path: str,
        pretrained_model_path: T.Optional[str] = None,
        num_epochs: int = 50, batch_size: int = 8,
        data_argu: bool = False, lr: float = 1e-4,
        summary_dir: str = "runs/unet",
        model_save_path: str = "best_unet_model.pth"
        ):
    """Train the UNet model.

    Args:
        meta_train_path: The path to the training metadata CSV file.
        meta_test_path: The path to the test metadata CSV file.
        dataset_root_path: The path to the root directory of the dataset.
        pretrained_model_path: The path to the pretrained model.
        num_epochs: The number of epochs to train for.
        batch_size: The batch size.
        data_argu: Whether to use data augmentation.
        lr: The learning rate.
        summary_dir: The directory to save the TensorBoard summary to.
        model_save_path: The path to save the best model to.
    """
    if data_argu:
        transform = Compose([
            RandomHorizontalFlip(),
            RandomRotation(),
            ToTensorWrapper(),
        ])
    else:
        transform = None

    train_dataset = FISHSpotsDataset(
        meta_csv=meta_train_path, root_dir=dataset_root_path,
        transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = FISHSpotsDataset(
        meta_csv=meta_test_path, root_dir=dataset_root_path)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(1, 1, 4)
    if pretrained_model_path is not None:
        print(f"Loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rmse_loss = RMSELoss()
    dice_loss = DiceLoss()

    def criterion(pred, target):
        loss_dice = dice_loss(pred, target)
        loss_rmse = rmse_loss(pred, target)
        return 0.6 * loss_dice + 0.4 * loss_rmse

    writer = SummaryWriter(summary_dir)
    train(
        model, optimizer, criterion, writer, device,
        train_loader, test_loader, model_save_path, num_epochs)


if __name__ == "__main__":
    fire.Fire(main)
