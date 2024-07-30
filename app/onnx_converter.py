""" Development code for FaceDetection
    Using RetinaFace from biubug6/Pytorch_Retinaface
    
    Development by bwchung
    
    Version 0.0.1 @2024.07.29
"""
from __future__ import print_function
import sys
from core.model_loader import ModelLoader
from core.parameters import RetinaFaceParameters
from core.detector import RetinaFaceDetector
from utils.etc import check_path

def onnx_convertion(config_file):
    """Executes detection"""
    if config_file.endswith(".json"):
        # Get model and convertion parameters
        model_params = RetinaFaceParameters()
        config_data = model_params.get_params(config_file)
        check_path(config_data["save_folder"])

        # Initializing modules
        net = ModelLoader(cfg=config_data, train=False)

        # Loading model
        print("Loading Model...")
        net.load_detection_model(config_data["trained_model"], config_data["cpu"])

        # Initialize RetinaFace
        print("Initializing detector...")
        detector = RetinaFaceDetector(net=net, cfg=config_data)

        # Export model to onnx
        detector.export_model(output_name="FaceDetector")

    else:
        print("Wrong format for train config file. Must be json format")
        print("Exiting app")
        sys.exit()
        