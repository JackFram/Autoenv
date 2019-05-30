from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord


class TemporalFeatureExtractor:
    def __init__(self):
        self.features = [0 for i in range(10)]
        self.num_features = 10

    def __len__(self):
        return self.num_features

    def pull_features(self, rec: SceneRecord, roadway: Roadway, veh_idx: int,
                      models: {}, pastframe: int = 0):
        idx = 0
        self.features[idx] = convert(Float64, get(
            ACC, rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert(Float64, get(
            JERK, rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert(Float64, get(
            TURNRATEG, rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert(Float64, get(
            ANGULARRATEG, rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert(Float64, get(
            TURNRATEF, rec, roadway, veh_idx, pastframe))
        idx += 1
        self.features[idx] = convert(Float64, get(
            ANGULARRATEF, rec, roadway, veh_idx, pastframe))

        # timegap is the time between when this vehicle's front bumper
        # will be in the position currently occupied by the vehicle
        # infront's back bumper
        timegap_censor_hi = 30.
        timegap = get(TIMEGAP, rec, roadway, veh_idx, pastframe, censor_hi=timegap_censor_hi)
        if timegap.v > timegap_censor_hi:
            timegap = FeatureValue(timegap_censor_hi, timegap.i)
        set_dual_feature!(self.features, idx, timegap, censor = timegap_censor_hi)
        idx += 2

        # inverse time to collision is the time until a collision
        # assuming that no actions are taken
        # inverse is taken so as to avoid infinite value, so flip here to get back
        # to TTC
        inv_ttc = get(INV_TTC, rec, roadway, veh_idx, pastframe)
        ttc = inverse_ttc_to_ttc(inv_ttc, censor_hi=30.0)
        set_dual_feature!(self.features, idx, ttc, censor = 30.0)
        return self.features


