from src.Roadway.roadway import Roadway, LaneTag, has_next, next_lane_point, next_lane, is_between_segments_hi, \
    is_between_segments_lo, is_in_entrances, is_in_exits
from src.Record.frame import Frame
from src.Vec import VecE2


class NeighborLongitudinalResult:
    def __init__(self, ind: int, delta_s: float):
        '''
        :param ind: index in scene of the neighbor
        :param delta_s: positive distance along lane between vehicles' positions
        '''

        self.ind = ind
        self.delta_s = delta_s


def get_neighbor_fore_along_lane_1(scene: Frame, roadway: Roadway, tag_start: LaneTag, s_base: float,
                                   targetpoint_primary: str, targetpoint_valid: str, max_distance_fore: float = 250.0,
                                   index_to_ignore: int = -1):

    # targetpoint_primary: the reference point whose distance we want to minimize
    # targetpoint_valid: the reference point, which if distance to is positive, we include the vehicle
    best_ind = None
    best_dist = max_distance_fore
    tag_target = tag_start

    dist_searched = 0.0
    while dist_searched < max_distance_fore:
        lane = roadway.get_by_tag(tag_target)
        for (i, veh) in enumerate(scene):
            if i != index_to_ignore:
                s_adjust = None
                if veh.state.posF.roadind.tag == tag_target:
                    s_adjust = 0.0
                elif is_between_segments_hi(veh.state.posF.roadind.ind, lane.curve) and \
                        is_in_entrances(roadway.get_by_tag(tag_target), veh.state.posF.roadind.tag):
                    distance_between_lanes = VecE2.norm(VecE2.VecE2(roadway.get_by_tag(tag_target).curve[0].pos -
                                                                    roadway.get_by_tag(veh.state.posF.roadind.tag).curve[-1].pos))
                    s_adjust = -(roadway.get_by_tag(veh.state.posF.roadind.tag).curve[-1].s + distance_between_lanes)
                elif is_between_segments_lo(veh.state.posF.roadind.ind) and \
                        is_in_exits(roadway.get_by_tag(tag_target), veh.state.posF.roadind.tag):
                    distance_between_lanes = VecE2.norm(VecE2.VecE2(roadway.get_by_tag(tag_target).curve[-1].pos -
                                                                    roadway.get_by_tag(veh.state.posF.roadind.tag).curve[0].pos))
                    s_adjust = roadway.get_by_tag(tag_target).curve[-1].s + distance_between_lanes
                if s_adjust is not None:
                    s_valid = veh.state.posF.s + veh.get_targetpoint_delta(targetpoint_valid) + s_adjust
                    dist_valid = s_valid - s_base + dist_searched
                    if dist_valid >= 0.0:
                        s_primary = veh.state.posF.s + veh.get_targetpoint_delta(targetpoint_primary) + s_adjust
                        dist = s_primary - s_base + dist_searched
                        if dist < best_dist:
                            best_dist = dist
                            best_ind = i

        if best_ind is not None:
            break
        if (not has_next(lane)) or (tag_target == tag_start and dist_searched != 0.0):
            # exit after visiting this lane a 2nd time
            break
        dist_searched += (lane.curve[-1].s - s_base)
        s_base = -VecE2.norm(VecE2.VecE2(lane.curve[-1].pos - next_lane_point(lane, roadway).pos))
        # negative distance between lanes
        tag_target = next_lane(lane, roadway).tag

    return NeighborLongitudinalResult(best_ind, best_dist)


def get_neighbor_fore_along_lane_2(scene: Frame, vehicle_index: int, roadway: Roadway, targetpoint_ego: str,
                                   targetpoint_primary: str, targetpoint_valid: str, max_distance_fore: float = 250.0):

    # targetpoint_primary: the reference point whose distance we want to minimize
    # targetpoint_valid: the reference point, which if distance to is positive, we include the vehicle

    veh_ego = scene[vehicle_index]
    tag_start = veh_ego.state.posF.roadind.tag
    s_base = veh_ego.state.posF.s + veh_ego.get_targetpoint_delta(targetpoint_ego)

    return get_neighbor_fore_along_lane_1(scene, roadway, tag_start, s_base,
                                          targetpoint_primary, targetpoint_valid,
                                          max_distance_fore=max_distance_fore, index_to_ignore=vehicle_index)


def get_neighbor_fore_along_lane_3(scene: Frame, vehicle_index: int, roadway: Roadway,
                                   max_distance_fore: float = 250.0):
    return get_neighbor_fore_along_lane_2(scene, vehicle_index, roadway, "Center", "Center", "Center",
                                          max_distance_fore=max_distance_fore)  # TODO: verify Center

