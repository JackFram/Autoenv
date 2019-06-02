from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord
from feature_extractor.interface import FeatureValue
from feature_extractor import FeatureState
import math


def get_LaneCurvature(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    veh = rec[pastframe][vehicle_index]
    curvept = roadway.get_by_id(veh.state.posF.roadind)
    val = curvept.k
    if val is None:
        return FeatureValue(0.0, FeatureState.MISSING)
    else:
        return FeatureValue(val)


def get_VelFs(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    veh = rec[pastframe][vehicle_index]
    return FeatureValue(veh.state.v*math.cos(veh.state.posF.phi))


def get_VelFt(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    veh = rec[pastframe][vehicle_index]
    return FeatureValue(veh.state.v*math.sin(veh.state.posF.phi))

