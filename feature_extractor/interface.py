from feature_extractor import FeatureState
from src.Record.record import SceneRecord, pastframe_inbounds, get_elapsed_time_3
from src.Roadway.roadway import Roadway
from feature_extractor import Get


def _get_feature_derivative_backwards(f: str, rec: SceneRecord, roadway: Roadway, vehicle_index: int,
                                      pastframe: int = 0, frames_back: int = 1):
    id = rec[pastframe][vehicle_index].id

    retval = FeatureState.FeatureValue(0.0, FeatureState.INSUF_HIST)
    pastframe2 = pastframe - frames_back

    if pastframe_inbounds(rec, pastframe) and pastframe_inbounds(rec, pastframe2):
        veh_index_curr = vehicle_index
        veh_index_prev = rec[pastframe2].findfirst(id)
        if veh_index_prev is not None:
            curr = None
            past = None
            if f == "VELFS":
                curr = convert_2_float(Get.get_VelFs(rec, roadway, veh_index_curr, pastframe))
                past = convert_2_float(Get.get_VelFs(rec, roadway, veh_index_prev, pastframe2))
            elif f == "SPEED":
                curr = convert_2_float(Get.get_Speed(rec, roadway, veh_index_curr, pastframe))
                past = convert_2_float(Get.get_Speed(rec, roadway, veh_index_prev, pastframe2))
            elif f == "ACC":
                curr = convert_2_float(get_Acc(rec, roadway, veh_index_curr, pastframe))
                past = convert_2_float(get_Acc(rec, roadway, veh_index_prev, pastframe2))
            elif f == "TURNRATEG":
                curr = convert_2_float(Get.get_TurnRateG(rec, roadway, veh_index_curr, pastframe))
                past = convert_2_float(Get.get_TurnRateG(rec, roadway, veh_index_prev, pastframe2))
            elif f == "POSFYAW":
                curr = convert_2_float(Get.get_PosFyaw(rec, roadway, veh_index_curr, pastframe))
                past = convert_2_float(Get.get_PosFyaw(rec, roadway, veh_index_prev, pastframe2))
            elif f == "TURNRATEF":
                curr = convert_2_float(Get.get_TurnRateF(rec, roadway, veh_index_curr, pastframe))
                past = convert_2_float(Get.get_TurnRateF(rec, roadway, veh_index_prev, pastframe2))
            else:
                raise ValueError("No matching feature as {}".format(f))
            delta_t = get_elapsed_time_3(rec, pastframe2, pastframe)
            retval = FeatureState.FeatureValue((curr - past) / delta_t)

    return retval


def convert_2_float(fv: FeatureState.FeatureValue):
    return fv.v


def get_AccFs(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("VELFS", rec, roadway, vehicle_index, pastframe)


def get_Acc(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("SPEED", rec, roadway, vehicle_index, pastframe)


def get_Jerk(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("ACC", rec, roadway, vehicle_index, pastframe)



