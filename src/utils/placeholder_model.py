import random

class PlaceholderModel:
    def __init__(self, config=None):
        pass

    def predict(self, video_path, question):
        return random.choice(['A', 'B', 'C', 'D'])
