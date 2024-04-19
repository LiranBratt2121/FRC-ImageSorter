from ultralytics import YOLO

from loader import FileLoader
from utils import process_image

def main():
    loader = FileLoader(clothes_model_name='clothing_model_best_yolov8m.pt',
                        input_dir='dis2-photos') # Change file for your needs

    print(f'Input images: {loader.input_dir}')

    clothes_model = YOLO(loader.clothes_detector_model_path, verbose=True)

    print(f'Clothes model: {clothes_model.model_name}')

    print('Starting to process images...')

    for image_path in loader.input_images_path:
        print(f'Processing image: {image_path}')
        process_image(image_path, clothes_model)
        print('Image processed')
    
    print('All images processed')

if __name__ == '__main__':
    main()
