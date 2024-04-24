from ultralytics import YOLO
import cv2

from loader import FileLoader
from utils import get_team

def main():
    loader = FileLoader(shirt_model_name='shirt_model_yolov8m-cls_best.pt',
                        input_dir='dis2-photos') # Change file for your needs

    print(f'Input images: {loader.input_dir}')

    shirt_model = YOLO(loader.shirt_detector_model_path, verbose=True)

    print('Starting to process images...')

    for image_path in loader.input_images_path:
        results = shirt_model.predict(source=image_path, conf=0.2)
        
        print(get_team(results))
        cv2.imshow('image', cv2.imread(image_path))
        cv2.waitKey(0)
        
    print('All images processed')

if __name__ == '__main__':
    main()
