from src.Record.record import SceneRecord
from src.Roadway.roadway import Roadway
from src.Basic import Vehicle
from feature_extractor.interface import convert_2_float
from feature_extractor.Get import get_Is_Colliding, get_RoadEdgeDist_Left, get_RoadEdgeDist_Right


class WellBehavedFeatureExtractor:
    def __init__(self):
        self.features = [0 for i in range(5)]
        self.num_features = 5

    def __len__(self):
        return self.num_features

    def pull_features(self, rec: SceneRecord, roadway: Roadway, veh_idx: int,
                      models: {}, pastframe: int = 0):
        scene = rec[pastframe]
        veh_ego = scene[veh_idx]
        d_ml = Vehicle.get_markerdist_left(veh_ego, roadway)
        d_mr = Vehicle.get_markerdist_right(veh_ego, roadway)
        idx = 0
        self.features[idx] = convert_2_float(get_Is_Colliding(rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = float(d_ml < -1.0 or d_mr < -1.0)
        idx += 1
        self.features[idx] = float(veh_ego.state.v < 0.0)
        idx += 1
        self.features[idx] = convert_2_float(get_RoadEdgeDist_Left(rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert_2_float(get_RoadEdgeDist_Right(rec, roadway, veh_idx, pastframe))
        return self.features



