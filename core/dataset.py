"""Module to prepare dataset for training"""
from data import preproc

class TrainDataset():
    """Train data creation class"""
    def __init__(self, training_dataset):
        self.training_dataset = training_dataset
        self.train_dataset_name = training_dataset.split("/")[1]

    def get_train_data(self, img_dim, rgb_mean):
        """Make training data"""
        if self.train_dataset_name == "widerface":
            from data import WiderFaceDetection as train_dataset
        else:
            pass
        train_data = train_dataset(self.training_dataset, preproc(img_dim, rgb_mean))
        return train_data
