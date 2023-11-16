import time
from enum import Enum
from typing import Dict, List

import matplotlib.cm as colormaps
import matplotlib.colors as mcolors
import matplotlib.markers as markers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.neural.data import TRANSFORMS, DataLoader, Mode


class ResidualBlock(nn.Module):
    """Residual block for ResNet."""

    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self, classes: List[str] = None, dropout_prob=0.5):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, len(classes)),
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        out = self.fc(features)
        return out, features


class ToDevice(Enum):
    """Device to use for the neural network. Allows for automatic detection of GPU or explicit use of CPU."""

    AUTO = "auto"
    CPU = "cpu"


class EvalType(Enum):
    """Type of evaluation to return."""

    PREDICTED = "predicted"
    CONFIDENCE_DICT = "confidence_dict"
    FEATURES = "features"


class Net:
    """Neural network for doodle prediction. Includes methods for training, evaluation, and visualization."""

    def __init__(
        self,
        loader: DataLoader,
        model_dir: str = "models",
        model_name: str = "doodle_prediction",
        to_device: ToDevice = ToDevice.AUTO,
    ):
        """Create a neural network.

        Args:
            loader (DataLoader): DataLoader object.
            model_dir (str, optional): Directory to save the model. Defaults to "models".
            model_name (str, optional): Name of the model. Defaults to "doodle_prediction".
            to_device (ToDevice, optional): Device to use for the neural network. Defaults to ToDevice.AUTO.
        """

        # Set the data loader
        self._loader = loader

        # Set the model directory and name
        self._model_dir = model_dir
        self._model_name = model_name

        # Create the neural network
        self._net = ResNet(loader.get_classes())

        # Set the device
        self._device = torch.device(
            "cuda:0"
            if torch.cuda.is_available() and to_device == ToDevice.AUTO
            else "cpu"
        )
        print(f"Using {self._device} for neural net.")
        self._net.to(self._device)

    def get_model_path(self, timestamp: bool = False) -> str:
        """Get the path to the model.

        Args:
            timestamp (bool, optional): Whether to include the timestamp. Defaults to False.

        Returns:
            str: Path to the model."""
        if timestamp:
            return f"{self._model_dir}/{self._model_name}_{int(time.time())}.pth"
        else:
            return f"{self._model_dir}/{self._model_name}.pth"

    def train(self, max_epochs: int = 60, patience: int = 6):
        """Train the neural network.

        Args:
            max_epochs (int, optional): Maximum number of epochs. Defaults to 100.
            patience (int, optional): Number of epochs with no improvement in validation accuracy before stopping. Also used for learning rate scheduler as max(1, patience // 2). Defaults to 6.
        """

        # Set the loss function, optimizer, and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self._net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=max(1, patience // 2), verbose=True
        )

        # Declare lists to store training and validation losses and accuracies
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # Parameters for early stopping
        best_val_accuracy = 0.0
        worse_epochs = 0

        # Get the train and validation loaders
        trainloader = self._loader.get_train_loader()
        valloader = self._loader.get_val_loader()

        # Train the network
        for epoch in range(max_epochs):
            running_loss = 0.0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader))

            # Train the network for the epoch
            for i, data in progress_bar:
                inputs, labels = data[0].to(self._device), data[1].to(
                    self._device
                )

                optimizer.zero_grad()

                outputs, features = self._net(inputs)
                _, predicted = torch.max(outputs, 1)
                total_correct += (
                    (predicted == labels).sum().item()
                )
                total_samples += labels.size(0)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_description(
                    f"Epoch {epoch+1} loss: {running_loss/(i+1):.3f} acc: {total_correct/total_samples:.3f}"
                )

            # Validate the network for the epoch
            self._net.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_total_correct = 0
                val_total_samples = 0

                progress_bar_val = tqdm(enumerate(valloader, 0), total=len(valloader))

                for i, data in progress_bar_val:
                    inputs, labels = data[0].to(self._device), data[1].to(
                        self._device
                    )

                    outputs, features = self._net(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_total_correct += (
                        (predicted == labels).sum().item()
                    )
                    val_total_samples += labels.size(0)

                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    progress_bar_val.set_description(
                        f"Validation loss: {val_loss/(i+1):.3f} acc: {val_total_correct/val_total_samples:.3f}"
                    )

                # Calculate the validation accuracy and append to the list
                val_epoch_accuracy = (
                    val_total_correct / val_total_samples
                ) 
                val_losses.append(val_loss / len(valloader))
                val_accuracies.append(
                    val_epoch_accuracy
                )

                # Save the model if the validation accuracy is better than the previous best else check for early stopping
                if val_epoch_accuracy > best_val_accuracy:
                    best_val_accuracy = val_epoch_accuracy
                    worse_epochs = 0
                    self.save()
                else:
                    worse_epochs += 1
                    if worse_epochs >= patience:
                        print("Early stopping")
                        break

                # Set the network back to training mode
                self._net.train()

            # Update the learning rate
            scheduler.step(val_epoch_accuracy)

            # Calculate the training accuracy and append to the list
            epoch_accuracy = (
                total_correct / total_samples
            )
            train_losses.append(running_loss / len(trainloader))
            train_accuracies.append(epoch_accuracy)
            running_loss = 0.0

        # Plot the training and validation loss and accuracy
        self.plot_training(train_losses, train_accuracies, val_losses, val_accuracies)
        print("Finished Training")

    def plot_training(
        self,
        train_losses: List[float],
        train_accuracies: List[float],
        val_losses: List[float],
        val_accuracies: List[float],
    ):
        """Plot the training and validation loss and accuracy.

        Args:
            train_losses (List[float]): Training losses.
            train_accuracies (List[float]): Training accuracies.
            val_losses (List[float]): Validation losses.
            val_accuracies (List[float]): Validation accuracies.
        """

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        plt.title("Training and Validation Loss over Time")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label="Training accuracy")
        plt.plot(val_accuracies, label="Validation accuracy")
        plt.title("Training and Validation Accuracy over Time")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate(
        self,
        input: torch.Tensor | Image.Image | List[Image.Image],
        return_type: EvalType = EvalType.PREDICTED,
        confidence_threshold: float = 0.05,
    ) -> torch.Tensor | Dict[str, float]:
        """Evaluate the neural network.

        Args:
            images (torch.Tensor | Image.Image | List[Image.Image]): Input images.
            return_type (EvalType, optional): Type of evaluation to return. Defaults to EvalType.PREDICTED.
            confidence_threshold (float, optional): Confidence threshold for confidence_dict. Defaults to 0.05.

        Returns:
            torch.Tensor | Dict[str, float]: Predicted classes or confidence dictionary.
        """
        self._net.eval()

        if isinstance(input, Image.Image):
            input = TRANSFORMS[Mode.EVAL](input)
            input = torch.unsqueeze(input, 0)
        elif isinstance(input, list):
            input = [TRANSFORMS[Mode.EVAL](img) for img in input]
            input = torch.stack(input)

        input = input.to(self._device)

        with torch.no_grad():
            outputs, features = self._net(input)
            if return_type == EvalType.CONFIDENCE_DICT:
                percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

                predictions = {}
                for i, c in enumerate(self._loader.get_classes()):
                    confidence = percentage[i].item() * 0.01

                    if confidence > confidence_threshold:
                        predictions[c] = confidence

                return predictions
            elif return_type == EvalType.FEATURES:
                return features
            else:
                _, predicted = torch.max(outputs, 1)
                return predicted

    def test_accuracy(self, by_class: bool = False, as_plot: bool = False):
        """Analyze the accuracy of the neural network against the test set.

        Args:
            by_class (bool, optional): Whether to return the accuracy for each class or the total accuracy. Defaults to False.
            as_plot (bool, optional): Whether to return the display as a plot or print to console. Defaults to False.
        """

        # Set the network to evaluation mode
        self._net.eval()

        if by_class:
            correct = {classname: 0 for classname in self._loader.get_classes()}
            total = {classname: 0 for classname in self._loader.get_classes()}
            
            # Iterate through the test set
            with torch.no_grad():
                for data in self._loader.get_test_loader():
                    images, labels = data
                    images, labels = data[0].to(self._device), data[1].to(self._device)
                    predicted = self.evaluate(images)

                    for label, prediction in zip(labels, predicted):
                        if label == prediction:
                            correct[self._loader.get_classes()[label]] += 1
                        total[self._loader.get_classes()[label]] += 1

            # Calculate the accuracy for each class
            accuracies = {}
            for classname, correct_count in correct.items():
                accuracies[classname] = 100 * float(correct_count) / total[classname]

            # Sort the classes by accuracy
            accuracies = sorted(accuracies.items(), key=lambda x: x[1])

            if as_plot:
                # Generate color map and normalize accuracy values for color bar
                cmap = colormaps.get_cmap(
                    "RdYlGn"
                )
                norm = mcolors.Normalize(
                    vmin=50, vmax=100
                )

                # Create the bar chart and axes
                fig, ax = plt.subplots(figsize=(10, 20))

                ax.barh(
                    range(len(accuracies)),
                    [x[1] for x in accuracies],
                    align="center",
                    color=[cmap(norm(x[1])) for x in accuracies],
                )

                # Add a color bar to indicate the mapping of accuracy to color
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(
                    sm, ax=ax, orientation="vertical"
                )
                cbar.set_label("Accuracy (%)")

                # Configure the plot labels and title
                ax.margins(y=0)
                ax.set_yticks(range(len(accuracies)))
                ax.set_yticklabels([x[0] for x in accuracies], fontsize=8)
                ax.set_title("Accuracy for Each Class", pad=10)
                ax.set_ylabel("Class")
                ax.set_xlabel("Accuracy (%)")

                # Show the plot
                plt.show()
            else:
                print(accuracies)
        else:
            correct = 0
            total = 0

            # Iterate through the test set
            with torch.no_grad():
                for data in self._loader.get_test_loader():
                    images, labels = data
                    images, labels = data[0].to(self._device), data[1].to(self._device)
                    predicted = self.evaluate(images)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            if as_plot:
                # Create the pie chart
                plt.figure(figsize=(6, 6))

                plt.pie(
                    [correct, total - correct],
                    labels=["Correct", "Incorrect"],
                    colors=["green", "red"],
                    autopct="%1.1f%%",
                    startangle=90,
                )

                plt.title("Total Accuracy Across All Classes", pad=10)
                plt.axis("equal")

                plt.show()
            else:
                print(
                    f"Accuracy % of the network on the {total} test images: {100 * correct / total}"
                )

    def visualize_features(self):
        """Visualize the feature representations of the neural network using t-SNE."""

        # Initialize lists to store features and labels
        features_list = []
        labels_list = []

        # Extract features and labels from the network
        with torch.no_grad():
            for data in self._loader.get_test_loader():
                images, labels = data
                images, labels = images.to(self._device), labels.to(self._device)
                features = self.evaluate(images, return_type=EvalType.FEATURES)

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        # Convert lists to NumPy arrays
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)

        # Normalize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        features_2d = tsne.fit_transform(features_scaled)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Generate color map with fewer unique colors
        colors = colormaps.tab10(np.linspace(0, 1, 10))

        # Get list of matplotlib markers
        all_markers = list(markers.MarkerStyle.markers.keys())
        usable_markers = [
            m for m in all_markers if isinstance(m, str) and m != " " and m != ""
        ]

        for i, label in enumerate(self._loader.get_classes()):
            # Get the color and marker for the class
            color = colors[i % 10]
            marker = usable_markers[
                (i // 10) % len(usable_markers)
            ]
            
            # Plot the class
            plt.scatter(
                features_2d[labels == i, 0],
                features_2d[labels == i, 1],
                color=color,
                label=label,
                alpha=0.4,
                marker=marker,
            )

        # Move the legend to the right of the plot
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=5)
        plt.title("t-SNE Visualization of Feature Representations")

        # Show the plot
        plt.show()

    def save(self, snapshot: bool = True):
        """Save the model.

        Args:
            snapshot (bool, optional): Adds a timestamp to the model name to prevent overwriting. Defaults to True.
        """
        torch.save(self._net.state_dict(), self.get_model_path())

        if snapshot:
            torch.save(self._net.state_dict(), self.get_model_path(timestamp=True))

    def load(self, path: str = None):
        """Load the model.

        Args:
            path (str, optional): Path to the model. Use the default path if None. Defaults to None.
        """
        if path is None:
            path = self.get_model_path()

        self._net.load_state_dict(torch.load(path))
