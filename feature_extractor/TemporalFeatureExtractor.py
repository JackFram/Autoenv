class TemporalFeatureExtractor:
    def __init__(self):
        self.features = [0 for i in range(10)]
        self.num_features = 10

    def __len__(self):
        return self.num_features

