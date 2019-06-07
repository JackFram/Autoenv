from src.Roadway.roadway import Roadway, proj_1, RoadIndex
from src.Record.record import SceneRecord, pastframe_inbounds, get_elapsed_time_3
from feature_extractor import FeatureState
from feature_extractor.interface import _get_feature_derivative_backwards, convert_2_float
from src.Vec.VecSE2 import deltaangle
from feature_extractor.neighbor_feature import NeighborLongitudinalResult
import math
from feature_extractor.collision_detection import CPAMemory, get_first_collision
from src.Vec import VecE2


def get_LaneCurvature(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    veh = rec[pastframe][vehicle_index]
    curvept = roadway.get_by_roadindex(veh.state.posF.roadind)
    val = curvept.k
    if val is None:
        return FeatureState.FeatureValue(0.0, FeatureState.MISSING)
    else:
        return FeatureState.FeatureValue(val)


def get_VelFs(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    veh = rec[pastframe][vehicle_index]
    return FeatureState.FeatureValue(veh.state.v*math.cos(veh.state.posF.phi))


def get_VelFt(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    veh = rec[pastframe][vehicle_index]
    return FeatureState.FeatureValue(veh.state.v*math.sin(veh.state.posF.phi))


def get_Speed(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return FeatureState.FeatureValue(rec[pastframe][vehicle_index].state.v)


def get_TurnRateG(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0, frames_back: int = 1):
    id = rec[pastframe][vehicle_index].id

    retval = FeatureState.FeatureValue(0.0, FeatureState.INSUF_HIST)
    pastframe2 = pastframe - frames_back
    if pastframe_inbounds(rec, pastframe2):
        veh_index_curr = vehicle_index
        veh_index_prev = rec[pastframe2].findfirst(id)

        if veh_index_prev is not None:
            curr = rec[pastframe][veh_index_curr].state.posG.theta
            past = rec[pastframe2][veh_index_prev].state.posG.theta
            delta_t = get_elapsed_time_3(rec, pastframe2, pastframe)
            retval = FeatureState.FeatureValue(deltaangle(past, curr) / delta_t)

    return retval


def get_AngularRateG(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("TURNRATEG", rec, roadway, vehicle_index, pastframe)


def get_PosFyaw(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return FeatureState.FeatureValue(rec[pastframe][vehicle_index].state.posF.phi)


def get_TurnRateF(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("POSFYAW",rec, roadway, vehicle_index, pastframe)


def get_AngularRateF(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("TURNRATEF", rec, roadway, vehicle_index, pastframe)


def get_TimeGap(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0,
                neighborfore: NeighborLongitudinalResult = None, censor_hi: float = 10.0):
    v = rec[pastframe][vehicle_index].state.v
    if v <= 0.0 or neighborfore.ind is None:
        return FeatureState.FeatureValue(censor_hi, FeatureState.CENSORED_HI)
    else:
        scene = rec[pastframe]
        len_ego = scene[vehicle_index].definition.length_
        len_oth = scene[neighborfore.ind].definition.length_
        delta_s = neighborfore.delta_s - len_ego / 2 - len_oth / 2
        if delta_s > 0:
            return FeatureState.FeatureValue(delta_s / v)
        else:
            return FeatureState.FeatureValue(0.0)  # collision!


def get_Inv_TTC(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0,
                neighborfore: NeighborLongitudinalResult = None, censor_hi: float = 10.0):
    if neighborfore.ind is None:
        return FeatureState.FeatureValue(0.0, FeatureState.MISSING)

    else:
        scene = rec[pastframe]

        veh_fore = scene[neighborfore.ind]
        veh_rear = scene[vehicle_index]

        len_ego = veh_fore.definition.length_
        len_oth = veh_rear.definition.length_

        delta_s = neighborfore.delta_s - len_ego / 2 - len_oth / 2
        delta_v = veh_fore.state.v - veh_rear.state.v

        if delta_s < 0.0:  # collision!
            return FeatureState.FeatureValue(censor_hi, FeatureState.CENSORED_HI)
        elif delta_v > 0.0:  # front car is pulling away
            return FeatureState.FeatureValue(0.0)
        else:
            f = - delta_v / delta_s
            if f > censor_hi:
                return FeatureState.FeatureValue(f, FeatureState.CENSORED_HI)
            else:
                return FeatureState.FeatureValue(f)


def get_Is_Colliding(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0,
                     mem: CPAMemory = CPAMemory()):
    scene = rec[pastframe]
    is_colliding = float(get_first_collision(scene, vehicle_index, mem).is_colliding)
    return FeatureState.FeatureValue(is_colliding)


def get_RoadEdgeDist_Left(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    veh = rec[pastframe][vehicle_index]
    offset = veh.state.posF.t
    footpoint = veh.get_footpoint
    seg = roadway.get_by_id(veh.state.posF.roadind.tag.segment)
    lane = seg.lanes[-1]
    roadproj = proj_1(footpoint, lane, roadway)
    curvept = roadway.get_by_roadindex(RoadIndex(roadproj.curveproj.ind, roadproj.tag))
    lane = roadway.get_by_tag(roadproj.tag)
    vec = curvept.pos - footpoint
    return FeatureState.FeatureValue(lane.width/2 + VecE2.norm(VecE2.VecE2(vec.x, vec.y)) - offset)


def get_RoadEdgeDist_Right(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    veh = rec[pastframe][vehicle_index]
    offset = veh.state.posF.t
    footpoint = veh.get_footpoint
    seg = roadway.get_by_id(veh.state.posF.roadind.tag.segment)
    lane = seg.lanes[0]
    roadproj = proj_1(footpoint, lane, roadway)
    curvept = roadway.get_by_roadindex(RoadIndex(roadproj.curveproj.ind, roadproj.tag))
    lane = roadway.get_by_tag(roadproj.tag)
    vec = curvept.pos - footpoint
    return FeatureState.FeatureValue(lane.width/2 + VecE2.norm(VecE2.VecE2(vec.x, vec.y)) + offset)






