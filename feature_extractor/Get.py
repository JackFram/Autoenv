from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord
from feature_extractor.interface import FeatureValue
from feature_extractor import FeatureState


def get_LaneCurvature(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    veh = rec[pastframe][vehicle_index]
    curvept = roadway[veh.state.posF.roadind]
    val = curvept.k
    if val is None:
        return FeatureValue(0.0, FeatureState.MISSING)
    else:
        return FeatureValue(val)


