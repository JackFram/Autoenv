class WellBehavedFeatureExtractor:
    def __init__(self):
        self.features = [0 for i in range(5)]
        self.num_features = 5

    def __len__(self):
        return self.num_features


