import cv2
from numpy import ndarray
import openpyxl
from openpyxl.drawing.image import Image

from typing import Dict, List

from dataclasses import dataclass

from utils import to_ROI


@dataclass
class Shirt:
    """
    A data structure representing a detected shirt.

    Attributes:
      team_name (str): Name of the team the shirt belongs to (without year and info).
      confidence (float): Confidence level of identifying the Shirt in the image (0.0 to 1.0).
      roi (ndarray): Region of interest (bounding box) for the member in the image.
      year (str | None): Year associated with the team (optional).
      other_data (str | None): Additional information about the team (optional).
    """
    team_name: str
    confidence: float
    roi: ndarray
    year: str = None
    other_data: str = None


class ImageData:
    """
    A data structure containing information about an image and detected shirts.

    Attributes:
      image_path (str): Path to the image.
      teams (Dict[str, List[Shirt]]): Dictionary mapping team names (without year and info) to a list of Shirt objects.
      team_occurrences (Dict[str, int]): Dictionary mapping team names (without year and info) to their occurrence count.

    """

    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.teams: Dict[str, List[Shirt]] = {}
        self.team_occurrences: Dict[str, int] = {}

    def __repr__(self) -> str:
        """
        Returns a string representation of the ImageData object.
        """
        return f'ImageData(image_path={self.image_path}, teams={self.teams}, team_occurrences={self.team_occurrences})'

    def add(self, team_name: str, confidence: float, roi: ndarray, year: str = None, other_data: str = None) -> None:
        """
        Adds a shirt to the image data with the specified information.

        Args:
        team_name (str): Name of the team the shirt belongs to (without year and info).
        confidence (float): Confidence level of identifying the shirt in the image (0.0 to 1.0).
        roi (ndarray): Region of interest (bounding box) for the shirt in the image.
        year (str, optional): Year associated with the team. Defaults to None.
        other_data (str, optional): Additional information about the shirt. Defaults to None.
        """
        if team_name not in self.teams:
            self.teams[team_name] = []
            self.team_occurrences[team_name] = 0
            return

        self.teams[team_name].append(Shirt(team_name, confidence, roi, year, other_data))
        self.team_occurrences[team_name] += 1

    @property
    def get_team_names(self) -> List[str]:
        """
        Returns a list of unique team names (without year and additional info).

        Returns:
          List[str]: List of team names.
        """
        return list(self.teams.keys())
    
