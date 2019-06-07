from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord
from feature_extractor import FeatureState


class MultiFeatureExtractor:
    def __init__(self, extractors: []):
        '''

        :param extractors: List{FeatureExtractors}
        '''
        self.extractors = extractors
        self.lengths = [len(subext) for subext in extractors]
        self.num_features = sum(self.lengths)
        self.features = [0 for i in range(self.num_features)]

    def __len__(self):
        return self.num_features

    def pull_features(self, rec: SceneRecord, roadway: Roadway, vehicle_index: int,
                      models: {}=dict(), pastframe: int = 0):  # = Dict{Int, DriverModel}()
        feature_index = 0
        for (subext, length) in zip(self.extractors, self.lengths):
            stop = feature_index + length
            self.features[feature_index:stop] = subext.pull_features(rec, roadway, vehicle_index, models, pastframe)
            feature_index += length
        return self.features

    def feature_names(self):
        fs = []
        for subext in self.extractors:
                for feature in subext.feature_names():
                    fs.append(feature)
        return fs

    def feature_info(self):
        info = dict()
        for subext in self.extractors:
            info = dict(**info, **(subext.feature_info()))
        return info


def set_feature_missing(features: list, i: int, censor: float = 0.):
    features[i] = censor
    features[i + 1] = 1.0
    return features


def set_feature(features: list, i: int, v: float):
    features[i] = v
    features[i + 1] = 0.0
    return features


def set_dual_feature(features: list, i: int, f: FeatureState.FeatureValue, censor: float = 0.):
    if f.i == FeatureState.MISSING:
        features = set_feature_missing(features, i, censor=censor)
    else:
        features = set_feature(features, i, f.v)

    return features



