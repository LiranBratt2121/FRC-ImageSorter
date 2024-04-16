from typing import Dict, List

class ImageData:
    """
    A data structure containing information about the image.
    
    Attributes:
        teams (List[Dict[str: float | str]]): List of teams in the image.
        [{
            "name": "1943",
            "conf": 0.5    
        }]

        image_path (str): Path to the image.
    """

    def __init__(self, image_path) -> None:
        '''
        Initialize the ImageData object.

        Args:
            image_path (str): Path to the target image.
        '''
        self.teams: List[Dict[str: float | str]] = []
        self.image_path = image_path

    def add(self, name: str, conf: float) -> None:
        """
        Adds a team to the teams list if it doesn't already exist.
        """
        if not any(team.get('name') == name for team in self.teams):
            self.teams.append({"name": name, "conf": conf})

    def __repr__(self) -> str:
        """
        Returns a string representation of the ImageData object.
        """
        return f"ImageData(image_path={self.image_path}, teams={self.teams})"
