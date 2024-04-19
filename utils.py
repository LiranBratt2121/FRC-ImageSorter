import torch
import numpy as np
from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Results
from typing import Dict, Tuple

def to_np_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Turns a torch tensor into a numpy array

    Args:
        tensor (torch.Tensor): A torch tensor

    Returns:
        np.ndarray: A numpy array representing the torch tensor.
    """
    return tensor.detach().cpu().numpy()

def to_ROI(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """
    Extracts a region of interest from an image.
    Uses xyxy format for the ROI.

    Args:
        img (np.ndarray): The image to extract the ROI from.
        roi (np.ndarray): The region of interest to extract.

    Returns:
        np.ndarray: The extracted region of interest.
    """
    x1, y1, x2, y2 = roi
    return img[int(y1): int(y2), int(x1): int(x2)]

def get_team(shirt_model_results: Results) -> Tuple[str, float]:
    """
    Gets the team name & confidence from the results of the shirt model.

    Args:
        shirt_model_results (Results): The results of the shirt model.

    Returns:
        A tupple containing the team name and the confidence score.
    """
    team_names: Dict = shirt_model_results[0].names
    team_in_photo_index = shirt_model_results[0].probs.top1

    return team_names[team_in_photo_index], round(float(shirt_model_results[0].probs.top1conf), 3)


def process_image(image_path: str, clothes_model: YOLO) -> None:
    """
    Processes a single image using the provided models.
    This function loads the image, calls the clothes detection model, and saves the results.

    Args:
        image_path (str): Path to the image.
        clothes_model (ultralytics.YOLO): The clothes detection model.

    Returns:
        None: This function performs processing and may print results.
    """

    try:
        image = cv2.imread(image_path)

        if image is None:
            print(f'Error loading image: {image_path}')
            return

        clothes_model.predict(image, verbose=True, save_crop=True, classes=[0, 1], conf=0.4)

        print(f'Successfully processed image: {image_path}')

    except Exception as e:
        print(f'Error processing image: {image_path} - {e}')