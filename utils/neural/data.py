"""Data utilities for the Doodle Zoo app."""

from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


class Mode(Enum):
    """Mode enum. Used to differentiate between training and evaluation modes."""

    TRAIN = "train"
    EVAL = "eval"


# Define the transforms globally for reusability
TRANSFORMS = {
    Mode.TRAIN: transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45, expand=True, fill=1),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    Mode.EVAL: transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
}


class DataLoader:
    """The DataLoader class provides utilities to split and load the dataset."""

    def __init__(
        self,
        data_dir: str = "data/base",
        data_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 100,
    ):
        """Load the dataset and split it into train, test, and validation sets.

        Args:
            data_dir (str, optional): Path to the dataset. Defaults to "data/training".
            data_split (Tuple(float, float, float), optional): Train, test, and validation split. Defaults to (0.8, 0.1, 0.1).
            batch_size (int, optional): Batch size. Defaults to 100."""

        # Ensure that the data split is valid
        if sum(data_split) != 1:
            raise ValueError("Data split must sum to 1")

        # Set the parameters
        self._data_dir = data_dir
        self._data_split = data_split
        self._batch_size = batch_size

        # Load the dataset
        self._full_dataset = torchvision.datasets.ImageFolder(root=data_dir)

        # Split the dataset
        split_generator = torch.Generator().manual_seed(
            42
        )
        train_split, test_split, val_split = data_split
        train_size = int(train_split * len(self._full_dataset))
        test_size = int(test_split * len(self._full_dataset))
        val_size = int(val_split * len(self._full_dataset))
        (
            self._train_dataset,
            self._test_dataset,
            self._val_dataset,
        ) = torch.utils.data.random_split(
            self._full_dataset,
            [train_size, test_size, val_size],
            generator=split_generator,
        )

        # Apply the appropriate transforms
        self._train_dataset.dataset.transform = TRANSFORMS[Mode.TRAIN]
        self._test_dataset.dataset.transform = TRANSFORMS[Mode.EVAL]
        self._val_dataset.dataset.transform = TRANSFORMS[Mode.EVAL]

        # Create the data loaders
        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset, batch_size=batch_size, shuffle=True
        )
        self._test_loader = torch.utils.data.DataLoader(
            self._test_dataset, batch_size=batch_size, shuffle=False
        )
        self._val_loader = torch.utils.data.DataLoader(
            self._val_dataset, batch_size=batch_size, shuffle=False
        )

    def get_data_dir(self) -> str:
        """Get the data directory.

        Returns:
            str: Data directory."""
        return self._data_dir

    def get_data_split(self) -> Tuple[float, float, float]:
        """Get the data split.

        Returns:
            Tuple(float, float, float): Train, test, and validation split."""
        return self._data_split

    def get_batch_size(self) -> int:
        """Get the batch size.

        Returns:
            int: Batch size."""
        return self._batch_size

    def get_classes(self) -> List[str]:
        """Get the classes.

        Returns:
            list: List of classes."""
        return self._full_dataset.classes

    def get_train_loader(self) -> torch.utils.data.DataLoader:
        """Get the train loader.

        Returns:
            torch.utils.data.DataLoader: Train loader."""
        return self._train_loader

    def get_test_loader(self) -> torch.utils.data.DataLoader:
        """Get the test loader.

        Returns:
            torch.utils.data.DataLoader: Test loader."""
        return self._test_loader

    def get_val_loader(self) -> torch.utils.data.DataLoader:
        """Get the validation loader.

        Returns:
            torch.utils.data.DataLoader: Validation loader."""
        return self._val_loader

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Get the data loaders.

        Returns:
            tuple: Tuple of train, test, and validation loaders."""
        return self._train_loader, self._test_loader, self._val_loader

    def get_train_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of training data.

        Returns:
            tuple: Tuple of images and labels."""
        return next(iter(self._train_loader))

    def get_test_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of test data.

        Returns:
            tuple: Tuple of images and labels."""
        return next(iter(self._test_loader))

    def get_val_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of validation data.

        Returns:
            tuple: Tuple of images and labels."""
        return next(iter(self._val_loader))


class DataVisualizer:
    """The DataVisualizer class provides utilities to visualize the dataset."""

    def __init__(self, data_loader: DataLoader):
        """Initialize the DataVisualizer.

        Args:
            data_loader (DataLoader): DataLoader object."""
        self._data_loader = data_loader

    def show_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor = None,
        predicted: torch.Tensor = None,
        n_rows: int = 10,
    ):
        """Show a batch of images.

        Args:
            images (torch.Tensor): Batch of images.
            labels (torch.Tensor): Batch of labels. Defaults to None.
            predicted (torch.Tensor): Batch of predictions. Defaults to None.
            n_rows (int, optional): Number of rows. Defaults to 10."""
        # Make a grid from batch
        out = torchvision.utils.make_grid(images, nrow=n_rows, normalize=True)

        # Plot the images
        plt.imshow(out.numpy().transpose((1, 2, 0)), cmap="gray")
        plt.axis("off")
        plt.show()

        # Print the ground truth labels
        if labels is not None:
            classes = self._data_loader.get_classes()
            row_len = len(images) // n_rows
            print(
                "Top Row:" if predicted is None else "Top Row (GroundTruth):",
                " | ".join(classes[labels[j]] for j in range(row_len)),
            )

        # Print the predictions if available
        if predicted is not None:
            print(
                "Top Row (Predictions):",
                " | ".join(classes[predicted[j]] for j in range(row_len)),
            )

    def show_random_train_batch(self):
        """Show a random batch of training images."""

        # Get some random training images
        images, labels = self._data_loader.get_train_batch()

        # Show images
        self.show_batch(images, labels)
