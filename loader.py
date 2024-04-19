import os

class FileLoader:
    """
    Loads files/images from a specified input directory and provides paths to output images.
    """
    
    def __init__(self,
                 input_dir="input_images",
                 output_dir="output_images",
                 models_dir='models',
                 clothes_model_name='clothes_detector_yolov8m.pt',
                 shirt_model_name='shirt_model_yolov8m-cls_last.pt'):
        """
        Initializes the ImageLoader with default input and output directory paths.

        Args:
            input_dir (str, optional): The directory containing input images.
                Defaults to "input_images".
            output_dir (str, optional): The directory to save output images.
                Defaults to "output_images".
            models_dir (str, optional): The directory to save the clothing models.
                Defaults to "models".
            clothes_model_name (str, optional): The name of the clothing model.
                Defaults to "clothes_detector_yolov8n".
            shirt_model_name (str, optional): The name of the shirt model.
                Defaults to "shirt_detector_yolov8n".
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.__models_dir = models_dir
        self.__clothes_model_name = clothes_model_name
        self.__shirt_model_name = shirt_model_name

    @property
    def input_images_path(self) -> list[str]:
        """
        Returns a list of paths to all files in the input directory.

        Raises:
            FileNotFoundError: If the input directory does not exist.
        """
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(
                f"Input directory not found: {self.input_dir}")

        return [os.path.join(self.input_dir, filename) for filename in os.listdir(self.input_dir)]

    @property
    def output_images_path(self) -> list[str]:
        """
        Returns a list of paths for potential output images in the output directory.

        (Paths do not necessarily correspond to actual images.)
        """

        # Create output directory if it doesn't exist
        os.mkdir(self.output_dir, exist_ok=True)
        return [os.path.join(self.output_dir, filename) for filename in os.listdir(self.output_dir)]

    @property
    def shirt_detector_model_path(self) -> str:
        """
        Returns path for the shirt_detector model.
        """
        return os.path.join(self.__models_dir, self.__shirt_model_name)

    @property
    def clothes_detector_model_path(self) -> str:
        """
        Returns path for the clothes_detector model.
        """
        return os.path.join(self.__models_dir, self.__clothes_model_name)
