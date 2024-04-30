from ultralytics import YOLO
from ultralytics.engine.results import Results

from typing import List
import cv2

from loader import FileLoader
from image_data import ImageData
from utils import parse_team, to_ROI, get_team, predict_images

loader = FileLoader(clothes_model_name='clothing_model_best_yolov8m.pt',
                    shirt_model_name='shirt_model_yolov8x-cls_last.pt',
                    input_dir='input_images')

clothes_model = YOLO(loader.clothes_detector_model_path)
shirt_model = YOLO(loader.shirt_detector_model_path)

preprocess_results = predict_images(
    model=clothes_model,
    imgs_path=loader.input_images_path,
    classes=[0, 1],
    conf=0.2)

data: List[ImageData] = []

for preprocess_result in preprocess_results:
    print(f'{preprocess_result=}')
    detections = preprocess_result[0].numpy().boxes.xyxy

    img = preprocess_result[0].orig_img
    image_data = ImageData(image_path=preprocess_result[0].path)

    for detection in detections:
        roi = to_ROI(img, detection)
        results: List[Results] = shirt_model.predict(source=roi, conf=0.2, verbose=False)
        # cv2.imshow('image', roi)
        # print(get_team(results))
        # cv2.waitKey(0)
        shirt_info = parse_team(get_team(results)[0])
        
        image_data.add(shirt_info.get('name'), get_team(results)[1], detection, shirt_info.get('year'), f"{shirt_info.get('orientation')}-{shirt_info.get('other_data')}")

    data.append(image_data)

ImageData.to_excel(data, 'output.xlsx')