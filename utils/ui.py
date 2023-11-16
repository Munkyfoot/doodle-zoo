"""The user interface module for the Doodle Zoo app."""

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from utils.db import Database
from utils.neural.data import DataLoader
from utils.neural.net import EvalType, Net


class Interface:
    """The user interface for the Doodle Zoo app."""

    def __init__(self):
        """Initialize and launch the interface."""

        # Create the database
        self._db = Database()

        # Create the neural network and load the model
        self._loader = DataLoader()
        self._net = Net(self._loader, to_device="cpu")
        self._net.load()

        # Create the app ui using Gradio
        self._app = None
        with gr.Blocks() as self._app:
            self._app.title = "The Doodle Zoo"
            self._app.css = """.preview {
            width: var(--size-full);
            }"""

            gr.Markdown(
                """# The Doodle Zoo
## A world of doodles at your finger tips!"""
            )

            # Create the "View Doodles" tab with a dropdown to select a category/label and a gallery to display the doodles
            with gr.Tab("View Doodles"):
                # Get the labels from the database - use only the base data labels
                labels = self._db.get_labels(True)

                # Create the dropdown to select a category/label
                inp_dropdown = gr.Dropdown(
                    choices=labels,
                    label="Pick a Category",
                    interactive=True,
                )

                # Create the gallery to display the doodles
                out_gallery = gr.Gallery(columns=5, label="Doodles")

                # Create the interface to allow the user to select a category/label and display the doodles
                gr.Interface(
                    fn=self._db.get_paths,
                    inputs=inp_dropdown,
                    outputs=out_gallery,
                    live=True,
                    allow_flagging="never",
                    api_name="display",
                )

                # Create a refresh button to refresh the gallery for the selected category/label
                refresh_button = gr.Button(value="Refresh Category")
                refresh_button.click(
                    fn=self._db.get_paths,
                    inputs=inp_dropdown,
                    outputs=out_gallery,
                    api_name="display_refresh",
                )

            # Create the "Create a Doodle" tab with a sketchpad to draw a doodle and an interface to submit the doodle and display the prediction
            with gr.Tab("Create a Doodle"):
                # Create the sketchpad to draw a doodle
                inp_sketchpad = gr.Sketchpad(
                    shape=(128, 128), label="Doodle", brush_radius=1
                )

                # Create the label to display the prediction
                out_label = gr.Label(label="Classification")

                # Create the interface to allow the user to submit the doodle and display the prediction
                gr.Interface(
                    fn=self.predict_doodle,
                    inputs=inp_sketchpad,
                    outputs=out_label,
                    allow_flagging="never",
                )

        # Launch the app
        self._app.launch()

    def process_doodle(self, doodle: np.ndarray) -> Image:
        """Process the doodle from the sketchpad to prepare it for classification by performing image transformations to match the base data - does not convert to tensor.

        Args:
            doodle (np.ndarray): The doodle as a numpy array.

        Returns:
            Image: The processed doodle as a PIL image."""

        # Convert to PIL image
        doodle_img = Image.fromarray(np.uint8(doodle))

        # Get bounding box
        bbox = doodle_img.getbbox()

        # Invert the image so that the doodle is black and the background is white
        doodle_img = ImageOps.invert(doodle_img)

        # If there is no bounding box, return the full image
        if not bbox:
            return doodle_img

        # Get the cropped doodle
        doodle_cropped = doodle_img.crop(bbox)

        # Get the dimensions and aspect ratio of the cropped doodle
        crop_width, crop_height = doodle_cropped.size
        crop_aspect = crop_height / crop_width

        # Resize the cropped doodle to fill a 128x128 image while maintaining the aspect ratio
        if crop_width > crop_height:
            new_width = 128
            new_height = int(128 * crop_aspect)
        else:
            new_height = 128
            new_width = int(new_height / crop_aspect)
        doodle_cropped = doodle_cropped.resize(
            (new_width, new_height), Image.Resampling.NEAREST
        )

        # Paste the cropped doodle onto a 128x128 white image
        doodle_img = Image.new("L", (128, 128), "white")
        doodle_img.paste(doodle_cropped, (0, 0))

        # Return the processed doodle
        return doodle_img

    def predict_doodle(
        self, doodle: np.ndarray, confidence_threshold: float = 0.8
    ) -> str | dict:
        """Get the predicted label(s) for the doodle.

        Args:
            doodle (np.ndarray): The doodle as a numpy array.
            confidence_threshold (float, optional): The confidence threshold for the prediction. Defaults to 0.8.

        Returns:
            Union[str, dict]: If the top prediction is above the confidence threshold, returns the predictions as a dictionary. If not, returns a message that the doodle could not be classified.
        """

        # Process the doodle to prepare it for classification
        doodle_img = self.process_doodle(doodle)

        # Get the predictions
        predictions = self._net.evaluate(doodle_img, EvalType.CONFIDENCE_DICT)

        # If there are no predictions, inform the user that the doodle could not be classified
        if predictions == {}:
            return "Unable to classify doodle."

        # Get the top prediction and its confidence
        top_prediction = max(predictions, key=predictions.get)
        top_prediction_confidence = predictions[top_prediction]

        # Check if the top prediction is above the confidence threshold
        if top_prediction_confidence > confidence_threshold:
            # If so, add the doodle to the database with the predicted label and return the predictions
            self._db.add_doodle(top_prediction, doodle_img)
            return predictions
        else:
            # If not, add the doodle to the database with the label "undetermined" and return a message that the doodle could not be classified
            self._db.add_doodle("undetermined", doodle_img)
            return "Unable to classify doodle."
