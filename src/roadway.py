# Zhihao Zhang
# NGSIM dataset processor roadway class

import re
import math
import os
import numpy as np
from src.Vec import VecE2, VecSE2
from src.curves import CurvePt
from src.Roadway import roadway
from src.splines import fit_cubic_spline, calc_curve_length_2, calc_curve_param_given_arclen, sample_spline_2, \
    sample_spline_theta_2, sample_spline_curvature_2, sample_spline_derivative_of_curvature_2

FLOATING_POINT_REGEX = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
METERS_PER_FOOT = 0.3048

DIR, filename = os.path.split(os.path.abspath(__file__))


class NGSIMRoadway:
    def __init__(self, name: str, boundaries: list, centerlines: list):
        self.name = name
        self.boundaries = boundaries
        self.centerlines = centerlines


class RoadwayInputParams:
    def __init__(self, filepath_boundaries: str, filepath_centerlines: str):
        self.filepath_boundaries = filepath_boundaries
        self.filepath_centerlines = filepath_centerlines


def read_boundaries(filepath_boundaries: str):
    fp = open(filepath_boundaries, 'r')
    lines = fp.readlines()
    fp.close()
    for i, line in enumerate(lines):
        lines[i] = line.strip()
    assert lines[0] == 'BOUNDARIES'

    n_boundaries = int(lines[1])

    assert n_boundaries >= 0

    retval = []  # Array{Vector{VecE2}}

    line_index = 1
    for i in range(n_boundaries):
        line_index += 1
        assert lines[line_index] == "BOUNDARY {}".format(i+1)
        line_index += 1
        npts = int(lines[line_index])
        line = []  # Array{VecE2}
        for j in range(npts):
            line_index += 1
            matches = re.findall(FLOATING_POINT_REGEX, lines[line_index])
            x = float(matches[0]) * METERS_PER_FOOT
            y = float(matches[1]) * METERS_PER_FOOT
            line.append(VecE2.VecE2(x, y))
        retval.append(line)
    return retval


def read_centerlines(filepath_centerlines: str):
    fp = open(filepath_centerlines, 'r')
    lines = fp.readlines()
    fp.close()
    for i, line in enumerate(lines):
        lines[i] = line.strip()
    assert lines[0] == 'CENTERLINES'
    n_centerlines = int(lines[1])
    assert n_centerlines >= 0
    line_index = 1
    retval = {}
    for i in range(n_centerlines):
        line_index += 1
        assert lines[line_index] == "CENTERLINE"
        line_index += 1
        name = lines[line_index]
        line_index += 1
        npts = int(lines[line_index])
        line = []
        for j in range(npts):
            line_index += 1
            matches = re.findall(FLOATING_POINT_REGEX, lines[line_index])
            x = float(matches[0]) * METERS_PER_FOOT
            y = float(matches[1]) * METERS_PER_FOOT
            line.append(VecE2.VecE2(x, y))

        centerline = []
        theta = (line[1] - line[0]).atan()
        centerline.append(CurvePt.CurvePt(VecSE2.VecSE2(line[0].x, line[0].y, theta), 0.0))
        for j in range(1, npts - 1):
            s = centerline[j - 1].s + (line[j] - line[j - 1]).hypot()
            theta = ((line[j] - line[j - 1]).atan() + (line[j + 1] - line[j]).atan())/2  # mean angle
            centerline.append(CurvePt.CurvePt(VecSE2.VecSE2(line[j].x, line[j].y, theta), s))
        s = centerline[npts - 2].s + (line[npts - 1] - line[npts - 2]).hypot()
        theta = (line[npts - 1] - line[npts - 2]).atan()
        centerline.append(CurvePt.CurvePt(VecSE2.VecSE2(line[npts - 1].x, line[npts - 1].y, theta), s))
        retval[name] = centerline
    return retval


def read_roadway(input_params: RoadwayInputParams):
    boundaries = read_boundaries(input_params.filepath_boundaries)
    centerlines = read_centerlines(input_params.filepath_centerlines)
    name = os.path.splitext(os.path.split('../data/centerlinesHOLO.txt')[1])[0]
    return NGSIMRoadway(name, boundaries, list(centerlines.values()))


def get_segid(lane: list):
    '''
    :param lane: list of CurvePt
    :return: get the segment id of corresponding lane
    '''
    return 1  # for now


def convert_curves_feet_to_meters(roadWay: roadway.Roadway):
    for seg in roadWay.segments:
        for lane in seg.lanes:
            for (i, P) in enumerate(lane.curve):
                lane.curve[i] = CurvePt.CurvePt(
                    VecSE2.VecSE2(P.pos.x * METERS_PER_FOOT, P.pos.y * METERS_PER_FOOT, P.pos.θ),
                    P.s * METERS_PER_FOOT, P.k / METERS_PER_FOOT, P.kd / METERS_PER_FOOT)

    return roadWay


def integrate(centerline_fn: str, boundary_fn: str,
              dist_threshold_lane_connect: float = 2.0,
              desired_distance_between_curve_samples: float = 1.0):
    '''
    :param centerline_path: center line file path
    :param boundary_path: boundary file path
    :param dist_threshold_lane_connect: [m]
    :param desired_distance_between_curve_samples: [m]
    :return:
    '''
    centerline_path = os.path.join(DIR, "../data/", centerline_fn)
    boundary_path = os.path.join(DIR, "../data/", boundary_fn)
    input_params = RoadwayInputParams(filepath_boundaries=boundary_path, filepath_centerlines=centerline_path)
    roadway_data = read_roadway(input_params)
    print("Finish loading centerlines and boundaries.")
    lane_pts_dict = dict()
    for (handle_int, lane) in enumerate(roadway_data.centerlines):
        segid = get_segid(lane)
        N = len(lane)
        pts = [None for _ in range(N)]  # VecE2 list
        for i in range(N):
            x = lane[i].pos.x
            y = lane[i].pos.y
            pts[i] = VecE2.VecE2(x, y)
        laneid = 1
        for tag in lane_pts_dict.keys():
            if tag.segment == segid:
                laneid += 1
        lane_pts_dict[roadway.LaneTag(segid, laneid)] = pts

    ###################################################
    # Shift pts to connect to previous / next pts

    lane_next_dict = dict()
    lane_prev_dict = dict()

    for (tag, pts) in lane_pts_dict.items():
        # see if can connect to next
        best_tag = roadway.NULL_LANETAG
        best_ind = -1
        best_sq_dist = dist_threshold_lane_connect
        for (tag2, pts2) in lane_pts_dict.items():
            if tag2.segment != tag.segment:
                for (ind, pt) in enumerate(pts2):
                    sq_dist = VecE2.normsquared(VecE2.VecE2(pt - pts[-1]))
                    if sq_dist < best_sq_dist:
                        best_sq_dist = sq_dist
                        best_ind = ind
                        best_tag = tag2
        if best_tag != roadway.NULL_LANETAG:
            # remove our last pt and set next to pt to their pt
            pts.pop()
            lane_next_dict[tag] = (lane_pts_dict[best_tag][best_ind], best_tag)
            if best_ind == 0:  # set connect prev as well
                lane_prev_dict[best_tag] = (pts[-1], tag)

    for (tag, pts) in lane_pts_dict.items():
        # see if can connect to prev
        if tag not in lane_prev_dict.keys():
            best_tag = roadway.NULL_LANETAG
            best_ind = -1
            best_sq_dist = dist_threshold_lane_connect
            for (tag2, pts2) in lane_pts_dict.items():
                if tag2.segment != tag.segment:
                    for (ind, pt) in enumerate(pts2):
                        sq_dist = VecE2.normsquared(VecE2.VecE2(pt - pts[0]))
                        if sq_dist < best_sq_dist:
                            best_sq_dist = sq_dist
                            best_ind = ind
                            best_tag = tag2
            if best_tag != roadway.NULL_LANETAG:
                lane_prev_dict[tag] = (lane_pts_dict[best_tag][best_ind], best_tag)

    ###################################################
    # Build the roadway
    retval = roadway.Roadway()
    for (tag, pts) in lane_pts_dict.items():
        if not retval.has_segment(tag.segment):
            retval.segments.append(roadway.RoadSegment(tag.segment))
    lane_new_dict = dict()  # old -> new tag
    for seg in retval.segments:

        # pull lanetags for this seg
        lanetags = []  # LaneTag
        for tag in lane_pts_dict.keys():
            if tag.segment == seg.id:
                lanetags.append(tag)

        # sort the lanes such that the rightmost lane is lane 1
        # do this by taking the first lane,
        # then project each lane's midpoint to the perpendicular at the midpoint

        assert len(lanetags) != 0
        proj_positions = [None for _ in range(len(lanetags))]  # list of float
        first_lane_pts = lane_pts_dict[lanetags[0]]
        n = len(first_lane_pts)
        lo = first_lane_pts[n//2 - 1]
        hi = first_lane_pts[n//2]
        midpt_orig = (lo + hi) / 2
        dir = VecE2.polar(1.0, (hi - lo).atan() + math.pi / 2)  # direction perpendicular (left) of lane

        for (i, tag) in enumerate(lanetags):
            pts = lane_pts_dict[tag]
            n = len(pts)
            midpt = (pts[n//2 - 1] + pts[n//2]) / 2
            proj_positions[i] = VecE2.proj_(midpt - midpt_orig, dir)

        for (i, j) in enumerate(sorted(range(len(proj_positions)), key=proj_positions.__getitem__)):
            tag = lanetags[j]

            boundary_left = roadway.LaneBoundary("solid", "white") if i == len(proj_positions) - 1 \
                else roadway.LaneBoundary("broken", "white")

            boundary_right = roadway.LaneBoundary("solid", "white") if i == 0 \
                else roadway.LaneBoundary("broken", "white")

            pts = lane_pts_dict[tag]
            pt_matrix = np.zeros((2, len(pts)))
            for (k, P) in enumerate(pts):
                pt_matrix[0, k] = P.x
                pt_matrix[1, k] = P.y
            print("fitting curve ", len(pts), "  ")
            curve = _fit_curve(pt_matrix, desired_distance_between_curve_samples)

            tag_new = roadway.LaneTag(seg.id, len(seg.lanes) + 1)
            lane = roadway.Lane(tag_new, curve,
                                boundary_left=boundary_left,
                                boundary_right=boundary_right)
            seg.lanes.append(lane)
            lane_new_dict[tag] = tag_new

    ###################################################
    # Connect the lanes
    for (tag_old, tup) in lane_next_dict.items():
        next_pt, next_tag_old = tup
        lane = retval.get_by_tag(lane_new_dict[tag_old])
        next_tag_new = lane_new_dict[next_tag_old]
        dest = retval.get_by_tag(next_tag_new)
        roadproj = roadway.proj_1(VecSE2.VecSE2(next_pt, 0.0), dest, retval)
        print("connecting {} to {}".format(lane.tag, dest.tag))
        cindS = CurvePt.curveindex_end(lane.curve)
        cindD = roadproj.curveproj.ind

        if cindD == CurvePt.CURVEINDEX_START:  # a standard connection
            lane, dest = roadway.connect(lane, dest)
            # remove any similar connection from lane_prev_dict
            if next_tag_old in lane_prev_dict.keys() and lane_prev_dict[next_tag_old][1] == tag_old:
                lane_prev_dict.pop(next_tag_old)
        else:
            lane.exits.insert(0, roadway.LaneConnection(True,  cindS, roadway.RoadIndex(cindD, dest.tag)))
            dest.entrances.append(roadway.LaneConnection(False, cindD, roadway.RoadIndex(cindS, lane.tag)))

    for (tag_old, tup) in lane_prev_dict.items():
        prev_pt, prev_tag_old = tup
        lane = retval.get_by_tag(lane_new_dict[tag_old])
        prev_tag_new = lane_new_dict[prev_tag_old]
        prev = retval.get_by_tag(prev_tag_new)
        roadproj = roadway.proj_1(VecSE2.VecSE2(prev_pt, 0.0), prev, retval)
        print("connecting {} from {}".format(lane.tag, prev.tag))
        cindS = roadproj.curveproj.ind
        cindD = CurvePt.CURVEINDEX_START
        if cindS == CurvePt.curveindex_end(prev.curve):  # a standard connection
            assert roadway.has_prev(prev)
            prev, lane = roadway.connect(prev, lane)
        else:
            # a standard connection
            prev.exits.append(roadway.LaneConnection(True,  cindS, roadway.RoadIndex(cindD, lane.tag)))
            lane.entrances.insert(0, roadway.LaneConnection(False, cindD, roadway.RoadIndex(cindS, prev.tag)))

    retval = convert_curves_feet_to_meters(retval)


def _fit_curve(pts, desired_distance_between_samples: float, max_iterations: int = 50,
               epsilon: float = 1e-4, n_intervals_in_arclen: int = 100):
    assert pts.shape[0] == 2
    spline_coeffs = fit_cubic_spline(pts)
    L = calc_curve_length_2(spline_coeffs[0], spline_coeffs[1], n_intervals_per_segment=n_intervals_in_arclen)
    n = round(L / desired_distance_between_samples) + 1
    s_arr = np.array([0.0 + i*L/n for i in range(n)])
    t_arr = calc_curve_param_given_arclen(spline_coeffs[0], spline_coeffs[1], s_arr,
                                          curve_length=L, max_iterations=max_iterations, epsilon=epsilon,
                                          n_intervals_in_arclen=n_intervals_in_arclen)

    x_arr = sample_spline_2(spline_coeffs[0], t_arr)
    y_arr = sample_spline_2(spline_coeffs[1], t_arr)
    theta_arr = sample_spline_theta_2(spline_coeffs[0], spline_coeffs[1], t_arr)
    print("Finish sampling theta")

    k_arr = sample_spline_curvature_2(spline_coeffs[0], spline_coeffs[1], t_arr)
    kd_arr = sample_spline_derivative_of_curvature_2(spline_coeffs[0], spline_coeffs[1], t_arr)

    print("Finish sampling curvature")

    # assert(!any(s->isnan(s), s_arr))
    # assert(!any(s->isnan(s), x_arr))
    # assert(!any(s->isnan(s), y_arr))
    # assert(!any(s->isnan(s), θ_arr))

    curve = [None for _ in range(n)]
    for i in range(n):
        pos = VecSE2.VecSE2(x_arr[i], y_arr[i], theta_arr[i])
        curve[i] = CurvePt.CurvePt(pos, s_arr[i], k_arr[i], kd_arr[i])

    return curve



