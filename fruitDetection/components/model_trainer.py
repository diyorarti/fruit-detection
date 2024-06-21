import os
import sys
import yaml
import glob
import shutil
from wasteDetection.utils.main_utils import read_yaml_file
from wasteDetection.logger import logging
from wasteDetection.exception import AppException
from wasteDetection.entity.config_entity import ModelTrainerConfig
from wasteDetection.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def get_class_names(self, label_dir: str):
        class_indices = set()
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        
        for label_file in label_files:
            with open(label_file, 'r') as file:
                for line in file:
                    class_idx = int(line.split()[0])
                    class_indices.add(class_idx)
        
        class_names = [f"class_{i}" for i in sorted(class_indices)]
        return class_names

    def create_data_yaml(self, image_dir: str, label_dir: str, output_file: str):
        images = glob.glob(os.path.join(image_dir, "*.jpg"))

        # Split the dataset into training and validation sets
        train_images = images[:int(len(images) * 0.8)]
        val_images = images[int(len(images) * 0.8):]

        # Write train and val files
        with open('train.txt', 'w') as f:
            for img in train_images:
                f.write(f"{os.path.abspath(img)}\n")
        
        with open('val.txt', 'w') as f:
            for img in val_images:
                f.write(f"{os.path.abspath(img)}\n")

        # Get class names from the label files
        class_names = self.get_class_names(label_dir)
        num_classes = len(class_names)

        # Write the data.yaml file
        data_yaml = {
            'train': os.path.abspath('train.txt'),
            'val': os.path.abspath('val.txt'),
            'nc': num_classes,
            'names': class_names
        }

        with open(output_file, 'w') as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=False)

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Creating data.yaml")
            self.create_data_yaml(
                'artifacts/data_ingestion/feature_store/Train File/Train File',
                'artifacts/data_ingestion/feature_store/Train File/Train File',
                'data.yaml'
            )

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")
            config['nc'] = len(self.get_class_names('artifacts/data_ingestion/feature_store/Train File/Train File'))  # Update with actual number of classes

            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            os.system(f"cd yolov5 && python train.py --img 416 --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --data ../data.yaml --cfg ./models/custom_{model_config_file_name}.yaml --weights {self.model_trainer_config.weight_name} --name yolov5s_results --cache")

            # Copy the best model to the desired directory
            shutil.copy("yolov5/runs/train/yolov5s_results/weights/best.pt", "yolov5/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy("yolov5/runs/train/yolov5s_results/weights/best.pt", os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt"))

            # Clean up
            shutil.rmtree("yolov5/runs")
            os.remove("train.txt")
            os.remove("val.txt")
            os.remove("data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)
