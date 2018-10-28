import os
import pickle

from train import TrainPipeline


def train_play():
    if os.path.exists('play.data'):
        with open('play.data', 'rb') as file:
            data = pickle.load(file)
            training_pipeline = TrainPipeline()
            training_pipeline.run(data)
