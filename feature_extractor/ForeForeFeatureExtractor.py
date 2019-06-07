from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord
from feature_extractor.neighbor_feature import get_neighbor_fore_along_lane_2, NeighborLongitudinalResult
from feature_extractor.interface import get_AccFs, convert_2_float


class ForeForeFeatureExtractor:
    def __init__(self, deltas_censor_hi: float = 100.):
        self.num_features = 3
        self.features = [0 for i in range(self.num_features)]
        self.deltas_censor_hi = deltas_censor_hi

    def __len__(self):
        return self.num_features

    def pull_features(self, rec: SceneRecord, roadway: Roadway, veh_idx: int,
                      models: {}, pastframe: int = 0):
        # reset features
        self.features = [0 for i in range(self.num_features)]

        scene = rec[pastframe]

        ego_vel = scene[veh_idx].state.v

        vtpf = "Front"
        vtpr = "Rear"
        fore_M = get_neighbor_fore_along_lane_2(scene, veh_idx, roadway, vtpf, vtpr, vtpf)
        if fore_M.ind is not None:
            fore_fore_M = get_neighbor_fore_along_lane_2(scene, fore_M.ind, roadway, vtpr, vtpf, vtpr)
        else:
            fore_fore_M = NeighborLongitudinalResult(0, 0.)

        if fore_fore_M.ind is not None:
            # total distance from ego vehicle
            self.features[0] = fore_fore_M.delta_s + fore_M.delta_s
            # relative velocity to ego vehicle
            self.features[1] = scene[fore_fore_M.ind].state.v - ego_vel
            # absolute acceleration
            self.features[2] = convert_2_float(get_AccFs(rec, roadway, fore_fore_M.ind, pastframe))
        else:
            self.features[0] = self.deltas_censor_hi
            self.features[1] = 0.
            self.features[2] = 0.

        return self.features

    def feature_names(self):
        return ["fore_fore_dist", "fore_fore_relative_vel", "fore_fore_accel"]

    def feature_info(self):
        return {
            "fore_fore_dist": {"high": 50., "low": 0},
            "fore_fore_relative_vel": {"high": 40., "low": -20.},
            "fore_fore_accel": {"high": 9., "low": -9.}
        }



