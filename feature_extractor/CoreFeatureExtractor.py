from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord


class CoreFeatureExtractor:
    def __init__(self):
        self.features = [0 for i in range(8)]
        self.num_features = 8

    def __len__(self):
        return self.num_features

    def pull_features(self, rec: SceneRecord, roadway: Roadway, veh_idx: int,
                      models: {}, pastframe: int = 0):
        scene = rec[pastframe]
        veh_ego = scene[veh_idx]
        d_ml = get_markerdist_left(veh_ego, roadway)
        d_mr = get_markerdist_right(veh_ego, roadway)
        idx = 0
        self.features[idx] = veh_ego.state.posF.t
        idx += 1
        self.features[idx] = veh_ego.state.posF.ϕ
        idx += 1
        self.features[idx] = veh_ego.state.v
        idx += 1
        self.features[idx] = veh_ego.def.length
        idx += 1
        self.features[idx] = veh_ego.def.width
        idx += 1
        self.features[idx] = convert(Float64, get(
            LANECURVATURE, rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = d_ml
        idx += 1
        self.features[idx] = d_mr
        return self.features