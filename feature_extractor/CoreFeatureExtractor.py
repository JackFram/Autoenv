class CoreFeatureExtractor:
    def __init__(self):
        self.features = [0 for i in range(8)]
        self.num_features = 8

    def __len__(self):
        return self.num_features

