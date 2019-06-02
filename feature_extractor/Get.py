from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord, pastframe_inbounds, get_elapsed_time_3
from feature_extractor.interface import FeatureValue, _get_feature_derivative_backwards
from feature_extractor import FeatureState
from src.Vec.geom.geom import deltaangle
from feature_extractor.neighbor_feature import NeighborLongitudinalResult
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


def get_Speed(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return FeatureValue(rec[pastframe][vehicle_index].state.v)


def get_TurnRateG(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0, frames_back: int = 1):
    id = rec[pastframe][vehicle_index].id

    retval = FeatureValue(0.0, FeatureState.INSUF_HIST)
    pastframe2 = pastframe - frames_back
    if pastframe_inbounds(rec, pastframe2):
        veh_index_curr = vehicle_index
        veh_index_prev = rec[pastframe2].findfirst(id)

        if veh_index_prev is not None:
            curr = rec[pastframe][veh_index_curr].state.posG.theta
            past = rec[pastframe2][veh_index_prev].state.posG.theta
            delta_t = get_elapsed_time_3(rec, pastframe2, pastframe)
            retval = FeatureValue(deltaangle(past, curr) / delta_t)

    return retval


def get_AngularRateG(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("TURNRATEG", rec, roadway, vehicle_index, pastframe)


def get_PosFyaw(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return FeatureValue(rec[pastframe][vehicle_index].state.posF.phi)


def get_TurnRateF(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("POSFYAW",rec, roadway, vehicle_index, pastframe)


def get_AngularRateF(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("TURNRATEF", rec, roadway, vehicle_index, pastframe)


def get_TimeGap(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0,
                neighborfore: NeighborLongitudinalResult = None, censor_hi: float = 10.0):
    v = rec[pastframe][vehicle_index].state.v
    if v <= 0.0 or neighborfore.ind is None:
        return FeatureValue(censor_hi, FeatureState.CENSORED_HI)
    else:
        scene = rec[pastframe]
        len_ego = scene[vehicle_index].definition.length_
        len_oth = scene[neighborfore.ind].definition.length_
        delta_s = neighborfore.delta_s - len_ego / 2 - len_oth / 2
        if delta_s > 0:
            return FeatureValue(delta_s / v)
        else:
            return FeatureValue(0.0)  # collision!


def get_Inv_TTC(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0,
                neighborfore: NeighborLongitudinalResult = None, censor_hi: float = 10.0):
    if neighborfore.ind is None:
        return FeatureValue(0.0, FeatureState.MISSING)

    else:
        scene = rec[pastframe]

        veh_fore = scene[neighborfore.ind]
        veh_rear = scene[vehicle_index]

        len_ego = veh_fore.definition.length_
        len_oth = veh_rear.definition.length_

        delta_s = neighborfore.delta_s - len_ego / 2 - len_oth / 2
        delta_v = veh_fore.state.v - veh_rear.state.v

        if delta_s < 0.0:  # collision!
            return FeatureValue(censor_hi, FeatureState.CENSORED_HI)
        elif delta_v > 0.0:  # front car is pulling away
            return FeatureValue(0.0)
        else:
            f = - delta_v / delta_s
            if f > censor_hi:
                return FeatureValue(f, FeatureState.CENSORED_HI)
            else:
                return FeatureValue(f)




