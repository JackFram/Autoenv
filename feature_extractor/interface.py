from feature_extractor import FeatureState
from src.Record.record import SceneRecord, pastframe_inbounds, get_elapsed_time_3
from src.Roadway.roadway import Roadway
from feature_extractor import Get


class FeatureValue:
    def __init__(self, v: float, i: int = FeatureState.GOOD):
        self.v = v
        self.i = i


def _get_feature_derivative_backwards(f: str, rec: SceneRecord, roadway: Roadway, vehicle_index: int,
                                      pastframe: int = 0, frames_back: int = 1):
    id = rec[pastframe][vehicle_index].id

    retval = FeatureValue(0.0, FeatureState.INSUF_HIST)
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
            else:
                raise ValueError("No matching feature as {}".format(f))
            delta_t = get_elapsed_time_3(rec, pastframe2, pastframe)
            retval = FeatureValue((curr - past) / delta_t)

    return retval


def convert_2_float(fv: FeatureValue):
    return fv.v


def get_AccFs(rec: SceneRecord, roadway: Roadway, vehicle_index: int, pastframe: int=0):
    return _get_feature_derivative_backwards("VELFS", rec, roadway, vehicle_index, pastframe)



