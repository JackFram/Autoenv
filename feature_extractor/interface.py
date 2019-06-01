from feature_extractor import FeatureState


class FeatureValue:
    def __init__(self, v: float, i: int = FeatureState.GOOD):
        self.v = v
        self.i = i

