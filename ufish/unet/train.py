import typing as T

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import Tensor

from .model import UNet
from .utils import DiceLoss, RMSELoss
from .data import FISHSpotsDataset
from ..utils.log import logger


def training_loop(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        criterion: T.Callable[[Tensor, Tensor], Tensor],
        writer: SummaryWriter,
        device: torch.device,
        train_loader: DataLoader, valid_loader: DataLoader,
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
        valid_loader: The valid data loader.
        model_save_path: The path to save the model to.
        num_epochs: The number of epochs to train for.
    """
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for idx, batch in enumerate(train_loader):
            images = batch["image"].to(device, dtype=torch.float)
            targets = batch["target"].to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                logger.info(
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
                    "Image/input",
                    images[0], epoch * len(train_loader) + idx)
                writer.add_image(
                    "Image/target",
                    targets[0], epoch * len(train_loader) + idx)
                writer.add_image(
                    "Image/pred",
                    outputs[0], epoch * len(train_loader) + idx)

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch["image"].to(device, dtype=torch.float)
                targets = batch["target"].to(device, dtype=torch.float)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Best model saved with Val Loss: {val_loss:.4f}")

    writer.close()


def train_on_dataset(
        train_dataset: FISHSpotsDataset,
        valid_dataset: FISHSpotsDataset,
        pretrained_model_path: T.Optional[str] = None,
        num_epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-4,
        summary_dir: str = "runs/unet",
        model_save_path: str = "best_unet_model.pth"
        ):
    """Train the UNet model.

    Args:
        train_dataset: The training dataset.
        valid_dataset: The validation dataset.
        pretrained_model_path: The path to the pretrained model.
        num_epochs: The number of epochs to train for.
        batch_size: The batch size.
        lr: The learning rate.
        summary_dir: The directory to save the TensorBoard summary to.
        model_save_path: The path to save the best model to.
    """

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = UNet(1, 1, 4)
    if pretrained_model_path is not None:
        logger.info(f"Loading pretrained model from {pretrained_model_path}")
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
    training_loop(
        model, optimizer, criterion, writer, device,
        train_loader, valid_loader, model_save_path, num_epochs)
