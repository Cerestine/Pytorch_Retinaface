""" Development code for FaceDetection
    Using RetinaFace from biubug6/Pytorch_Retinaface
    
    Development by bwchung
    
    Version 0.0.1 @2024.07.29
"""
from __future__ import print_function
import sys
from core.model_loader import ModelLoader
from core.parameters import RetinaFaceParameters
from core.trainer import RetinaFaceTrainer
from core.dataset import TrainDataset
from utils.etc import check_path

def train(config_file):
    """Executes training"""
    if config_file.endswith(".json"):
        # Get model and train parameters
        model_params = RetinaFaceParameters()
        config_data = model_params.get_params(config_file)
        check_path(config_data["save_folder"])

        # Initializing modules
        net = ModelLoader(cfg=config_data)
        dataset = TrainDataset(training_dataset=config_data["training_dataset"])

        # Load model
        print("Loading Model...")
        net.load_train_model(config_data["network"], config_data["num_gpu"], config_data["gpu_train"])

        # Load train dataset
        print("Loading Dataset...")
        train_data = dataset.get_train_data(config_data["img_dim"], config_data["rgb_mean"])

        # Initialize RetinaFace
        print("Initializing Trainer...")
        trainer = RetinaFaceTrainer(net=net, model_params=model_params, config_data=config_data)

        # Start training
        print("Start training...")
        trainer.train(config_data["resume_epoch"], train_data)

    else:
        print("Wrong format for train config file. Must be json format")
        print("Exiting app")
        sys.exit()
        