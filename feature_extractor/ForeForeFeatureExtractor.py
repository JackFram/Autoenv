class ForeForeFeatureExtractor:
    def __init__(self, deltas_censor_hi: float = 100.):
        self.num_features = 3
        self.features = [0 for i in range(self.num_features)]
        self.deltas_censor_hi = deltas_censor_hi

    def __len__(self):
        return self.num_features


