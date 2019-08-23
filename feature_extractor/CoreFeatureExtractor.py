from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord
from src.Basic import Vehicle
from feature_extractor.Get import get_LaneCurvature

'''
"relative_offset": vehicle's lane offset, positive is to left. zero point is the centerline of the lane. 
"relative_heading": the angle of the lane relative heading 
"velocity": velocity of the car
"length": length of the car
"width": width of the car
"lane_curvature": the curvature of the current lane point
"markerdist_left": the distance between the car and the left side of current lane
"markerdist_right": the distance between the car and the right side of current lane
'''


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
        d_ml = Vehicle.get_markerdist_left(veh_ego, roadway)
        d_mr = Vehicle.get_markerdist_right(veh_ego, roadway)
        idx = 0
        self.features[idx] = veh_ego.state.posF.t
        idx += 1
        self.features[idx] = veh_ego.state.posF.phi
        idx += 1
        self.features[idx] = veh_ego.state.v
        idx += 1
        self.features[idx] = veh_ego.definition.length_
        idx += 1
        self.features[idx] = veh_ego.definition.width_
        idx += 1
        self.features[idx] = get_LaneCurvature(rec, roadway, veh_idx, pastframe).v
        idx += 1
        self.features[idx] = d_ml
        idx += 1
        self.features[idx] = d_mr
        return self.features

    def feature_names(self):
        return ["relative_offset", "relative_heading", "velocity", "length",
                "width", "lane_curvature", "markerdist_left", "markerdist_right"]

    def feature_info(self):
        return {
            "relative_offset": {"high": 1., "low": -1.},
            "relative_heading": {"high": .05, "low": -.05},
            "velocity": {"high": 40., "low": -5.},
            "length": {"high": 30., "low": 2.},
            "width": {"high": 3., "low": .9},
            "lane_curvature": {"high": .1, "low": -.1},
            "markerdist_left": {"high": 3., "low": 0.},
            "markerdist_right": {"high": 3., "low": 0.}
        }

