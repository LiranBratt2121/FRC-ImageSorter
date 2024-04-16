from ultralytics import YOLO
from ultralytics.engine.results import Results

from typing import List

from loader import FileLoader
from image_data import ImageData
from utils import to_np_array, to_ROI, get_team

loader = FileLoader(clothes_model_name='clothing_model_best_yolov8m.pt',
                    shirt_model_name='shirt_model_yolov8m-cls_last.pt')

clothes_model = YOLO(loader.clothes_detector_model_path)
shirt_model = YOLO(loader.shirt_detector_model_path)

preprocess_results: List[Results] = clothes_model.predict(source=loader.input_images_path,
                                               stream=True,
                                               classes=[0, 1],
                                               conf=0.4)
data: List[ImageData] = []

for preprocess_result in preprocess_results:
    detections = [to_np_array(tensor.xyxy) for tensor in preprocess_result.boxes]

    img = preprocess_result.orig_img
    image_data = ImageData(image_path=preprocess_result.path)
    
    for detection in detections:
        roi = to_ROI(img, detection[0])
        results: List[Results] = shirt_model.predict(source=roi, conf=0.2)

        image_data.add(*get_team(results.copy()))

    data.append(image_data)