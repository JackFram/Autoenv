from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord
from feature_extractor.interface import convert_2_float, get_Acc, get_Jerk
from feature_extractor.Get import get_TurnRateG, get_AngularRateG, get_TurnRateF, get_AngularRateF, get_TimeGap, \
    get_Inv_TTC
from feature_extractor.neighbor_feature import get_neighbor_fore_along_lane_3
from feature_extractor.feature_extractor import set_dual_feature
from feature_extractor import FeatureState
'''
"accel": vehicle's acceleration 
"jerk": the derivative of teh accel
"turn_rate_global": global turn rate of the vehicle
"angular_rate_global": derivative of the global turn rate
"turn_rate_frenet": relative turn rate compared to lane
"angular_rate_frenet": derivative of turn rate compared to lane
"timegap": time gap of two consecutive steps
"timegap_is_avail": boolean var to indicate time gap information is available or not 
"time_to_collision": the time our vehicle will hit the front vehicle
"time_to_collision_is_avail": time to collision var information is available or not
'''


class TemporalFeatureExtractor:
    def __init__(self):
        self.features = [0 for i in range(10)]
        self.num_features = 10

    def __len__(self):
        return self.num_features

    def pull_features(self, rec: SceneRecord, roadway: Roadway, veh_idx: int,
                      models: {}, pastframe: int = 0):
        idx = 0
        self.features[idx] = convert_2_float(get_Acc(rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert_2_float(get_Jerk(rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert_2_float(get_TurnRateG(rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert_2_float(get_AngularRateG(rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert_2_float(get_TurnRateF(rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert_2_float(get_AngularRateF(rec, roadway, veh_idx, pastframe))

        # timegap is the time between when this vehicle's front bumper
        # will be in the position currently occupied by the vehicle
        # infront's back bumper
        timegap_censor_hi = 30.
        neighborfore = get_neighbor_fore_along_lane_3(rec[pastframe], veh_idx, roadway)
        timegap = get_TimeGap(rec, roadway, veh_idx, pastframe, neighborfore=neighborfore, censor_hi=timegap_censor_hi)
        if timegap.v > timegap_censor_hi:
            timegap = FeatureState.FeatureValue(timegap_censor_hi, timegap.i)
        self.features = set_dual_feature(self.features, idx, timegap, censor=timegap_censor_hi)
        idx += 2

        # inverse time to collision is the time until a collision
        # assuming that no actions are taken
        # inverse is taken so as to avoid infinite value, so flip here to get back
        # to TTC
        neighborfore = get_neighbor_fore_along_lane_3(rec[pastframe], veh_idx, roadway)
        inv_ttc = get_Inv_TTC(rec, roadway, veh_idx, pastframe, neighborfore=neighborfore)
        ttc = FeatureState.inverse_ttc_to_ttc(inv_ttc, censor_hi=30.0)
        self.features = set_dual_feature(self.features, idx, ttc, censor=30.0)
        return self.features

    def feature_names(self):
        return ["accel", "jerk", "turn_rate_global", "angular_rate_global",
                "turn_rate_frenet", "angular_rate_frenet",
                "timegap", "timegap_is_avail",
                "time_to_collision", "time_to_collision_is_avail"]

    def feature_info(self):
        return {
            "accel": {"high": 9., "low": -9.},
            "jerk": {"high": 70., "low": -70.},
            "turn_rate_global": {"high": .5, "low": -.5},
            "angular_rate_global": {"high": 3., "low": -3.},
            "turn_rate_frenet": {"high": .1, "low": -.1},
            "angular_rate_frenet": {"high": 3., "low": -3.},
            "timegap": {"high": 30., "low": 0.},
            "timegap_is_avail": {"high": 1., "low": 0.},
            "time_to_collision": {"high": 30., "low": 0.},
            "time_to_collision_is_avail": {"high": 1., "low": 0.}
        }


