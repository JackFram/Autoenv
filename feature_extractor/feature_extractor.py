from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord


class MultiFeatureExtractor:
    def __init__(self, extractors: []):
        '''

        :param extractors: List{FeatureExtractors}
        '''
        self.extractors = extractors
        self.lengths = [len(subext) for subext in extractors]
        self.num_features = sum(self.lengths)
        self.features = [0 for i in range(self.num_features)]

    def pull_features(self, rec: SceneRecord, roadway: Roadway, vehicle_index: int,
                      models: {}, pastframe: int = 0):  # = Dict{Int, DriverModel}()
        feature_index = 0
        for (subext, length) in zip(self.extractors, self.lengths):
            stop = feature_index + length
            self.features[feature_index:stop] = subext.pull_features(rec, roadway, vehicle_index, models, pastframe)
            feature_index += length
        return self.features

