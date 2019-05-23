# Zhihao Zhang
# roadway class in python
from curves import CurvePt
from Vec import VecSE2, VecE2
from Vec.geom import geom
import math
import re
import warnings


class LaneBoundary:
    def __init__(self, style: str, color: str):
        self.style = style
        self.color = color


class SpeedLimit:
    def __init__(self, lo: float, hi: float):
        self.lo = lo
        self.hi = hi

NULL_BOUNDARY = LaneBoundary("unknown", "unknown")
DEFAULT_SPEED_LIMIT = SpeedLimit(-math.inf, math.inf)
DEFAULT_LANE_WIDTH = 3.0


class LaneTag:
    def __init__(self, segment: int, lane: int):
        self.segment = segment
        self.lane = lane


NULL_LANETAG = LaneTag(0,0)


class RoadIndex:
    def __init__(self, ind: CurvePt.CurveIndex, tag: LaneTag):
        self.ind = ind
        self.tag = tag


NULL_ROADINDEX = RoadIndex(CurvePt.CurveIndex(-1, None), LaneTag(-1,-1))


class LaneConnection:
    def __init__(self, downstream: bool, mylane: CurvePt.CurveIndex, target: RoadIndex):
        self.downstream = downstream
        self.mylane = mylane
        self.target = target


def parse_lane_connection(line: str):
    cleanedline = re.sub(r"(\(|\))", "", line)
    print(cleanedline)
    tokens = cleanedline.split()
    assert tokens[0] == "D" or tokens[0] == "U"
    downstream = (tokens[0] == "D")
    mylane = CurvePt.CurveIndex(int(tokens[1]), float(tokens[2]))
    target = RoadIndex(
                CurvePt.CurveIndex(int(tokens[3]), float(tokens[4])),
                LaneTag(int(tokens[5]), int(tokens[6]))
            )
    return LaneConnection(downstream, mylane, target)


class Lane:
    def __init__(self, tag: LaneTag, curve: list, width: float = DEFAULT_LANE_WIDTH, speed_limit: SpeedLimit = DEFAULT_SPEED_LIMIT,
                 boundary_left: LaneBoundary = NULL_BOUNDARY, boundary_right: LaneBoundary = NULL_BOUNDARY, exits: list = [], entrances: list = [],
                 next: RoadIndex = NULL_ROADINDEX, prev: RoadIndex = NULL_ROADINDEX):
        self.tag = tag
        self.curve = curve  # Array of Curve
        self.width = width
        self.speed_limit = speed_limit
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.exits = exits  # Array of LaneConnection
        self.entrances = entrances  # Array of LaneConnection

        if next != NULL_ROADINDEX:
            self.exits.insert(0, LaneConnection(True, CurvePt.curveindex_end(self.curve), next))

        if prev != NULL_ROADINDEX:
            self.entrances.insert(0, LaneConnection(False, CurvePt.CURVEINDEX_START, prev))

    def get_by_ind_roadway(self, ind: CurvePt.CurveIndex, roadway):
        if ind.i == -1:
            pt_lo = prev_lane_point(self, roadway)
            pt_hi = self.curve[0]
            s_gap = VecE2.norm(VecE2.VecE2(pt_hi.pos - pt_lo.pos))
            pt_lo = CurvePt.CurvePt(pt_lo.pos, -s_gap, pt_lo.k, pt_lo.kd)
            return CurvePt.lerp(pt_lo, pt_hi, ind.t)
        elif ind.i < len(self.curve) - 1:
            return CurvePt.get_curve_list_by_index(self.curve, ind)
        else:
            pt_hi = next_lane_point(self, roadway)
            pt_lo = self.curve[-1]
            s_gap = VecE2.norm(VecE2.VecE2(pt_hi.pos - pt_lo.pos))
            pt_hi = CurvePt.CurvePt(pt_hi.pos, pt_lo.s + s_gap, pt_hi.k, pt_hi.kd)
            return CurvePt.lerp(pt_lo, pt_hi, ind.t)


def has_next(lane: Lane):
    return (not len(lane.exits) == 0) and lane.exits[0].mylane == CurvePt.curveindex_end(lane.curve)


def has_prev(lane: Lane):
    return (not len(lane.entrances) == 0) and lane.entrances[0].mylane == CurvePt.CURVEINDEX_START


class RoadSegment:
    def __init__(self, id: int, lanes: list):
        self.id = id  # integer
        self.lanes = lanes  # Array of Lane


class Roadway:
    def __init__(self, segments: list = []):
        self.segments = segments  # Array of RoadSegment

    def get_by_tag(self, tag: LaneTag):
        seg = self.get_by_id(tag.segment)
        #print(seg.id, len(seg.lanes), tag.lane)
        return seg.lanes[tag.lane]

    def get_by_id(self, segid: int):
        for seg in self.segments:
            if seg.id == segid:
                return seg
        raise IndexError("Could not find segid {} in roadway".format(segid))

    def get_by_roadindex(self, roadindex: RoadIndex):
        lane = self.get_by_tag(roadindex.tag)
        return lane.get_by_ind_roadway(roadindex.ind, self)


def read_roadway(fp):
    lines = fp.readlines()
    fp.close()
    line_index = 0
    if "ROADWAY" in lines[line_index]:  #文件第一行 ROADWAY title
        line_index += 1  #继续读下一行

    nsegs = int(lines[line_index].strip())  # 读Roadway.segments的长度
    line_index += 1  #继续读下一行

    roadway = Roadway([])
    for i_seg in range(nsegs):  # 循环读每个segments的内容
        segid = int(lines[line_index].strip())  # parse segments.id
        line_index += 1  #继续读下一行
        nlanes = int(lines[line_index].strip())  # 读出Roadway.segments.lanes的长度
        line_index += 1  #继续读下一行
        seg = RoadSegment(segid, [])  #
        for i_lane in range(nlanes):  # 循环读每个lane的内容
            assert i_lane + 1 == int(lines[line_index].strip())  # 这里是用来确认lane符合顺序且读的方式正确
            line_index += 1  #继续读下一行
            tag = LaneTag(segid, i_lane)  # make Roadway.segments.lanes.tag
            #print(segid, nlanes, i_lane)
            width = float(lines[line_index].strip())  # parse width
            line_index += 1  #继续读下一行
            tokens = (lines[line_index].strip()).split()
            line_index += 1  #继续读下一行
            speed_limit = SpeedLimit(float(tokens[0]), float(tokens[1]))  # parse speed limit
            tokens = (lines[line_index].strip()).split()
            line_index += 1  #继续读下一行
            boundary_left = LaneBoundary(tokens[0], tokens[1])  # parse boundary_left
            tokens = (lines[line_index].strip()).split()
            line_index += 1  #继续读下一行
            boundary_right = LaneBoundary(tokens[0], tokens[1])  # parse boundary_right
            exits = []
            entrances = []
            n_conns = int(lines[line_index].strip())  # 这里应该是Roadway.segments.lanes.exits以及entrances的长度
            line_index += 1  #继续读下一行
            for i_conn in range(n_conns):  # 循环parse每个exit以及entrance
                conn = parse_lane_connection(lines[line_index].strip())  # parse LaneConnection
                line_index += 1  #继续读下一行
                if conn.downstream:
                    exits.append(conn)  # if downstream 就是exit的数据
                else:
                    entrances.append(conn)  # else entrance的数据
            npts = int(lines[line_index].strip())  # 有多少个curve点————Roadway.segments.lanes.curve的长度
            line_index += 1  #继续读下一行
            curve = []
            for i_pt in range(npts):  # 循环parse CurvePt
                line = lines[line_index].strip()
                line_index += 1  #继续读下一行
                cleanedline = re.sub(r"(\(|\))", "", line)
                tokens = cleanedline.split()
                x = float(tokens[0])
                y = float(tokens[1])
                theta = float(tokens[2])
                s = float(tokens[3])
                k = float(tokens[4])
                kd = float(tokens[5])
                curve.append(CurvePt.CurvePt(VecSE2.VecSE2(x, y, theta), s, k, kd))  # append CurvePt in Roadway.segments.lanes.curve
            seg.lanes.append(Lane(tag, curve, width=width, speed_limit=speed_limit,
                                     boundary_left=boundary_left,
                                     boundary_right=boundary_right,
                                     entrances=entrances, exits=exits)) # append lane in Roadway.segments.lanes
        roadway.segments.append(seg)  # append segs in Roadway.segments
        #print(len(roadway.segments))
    return roadway


class RoadProjection:
    def __init__(self, curveproj: CurvePt.CurveProjection, tag: LaneTag):
        self.curveproj = curveproj
        self.tag = tag


def next_lane(lane: Lane, roadway: Roadway):
    return roadway.get_by_tag(lane.exits[0].target.tag)


def prev_lane(lane: Lane, roadway: Roadway):
    return roadway.get_by_tag(lane.entrances[0].target.tag)


def next_lane_point(lane: Lane, roadway: Roadway):
    return roadway.get_by_roadindex(lane.exits[0].target)


def prev_lane_point(lane: Lane, roadway: Roadway):
    return roadway.get_by_roadindex(lane.entrances[0].target)


def sign(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0


def get_closest_perpendicular_point_between_points(A: VecSE2.VecSE2, B: VecSE2.VecSE2, Q: VecSE2.VecSE2,
    tolerance: float = 0.01, # acceptable error in perpendicular component
    max_iter: int = 50, # maximum number of iterations
    ):

    # CONDITIONS: a < b, either f(a) < 0 and f(b) > 0 or f(a) > 0 and f(b) < 0
    # OUTPUT: value which differs from a root of f(x)=0 by less than TOL

    a = 0.0
    b = 1.0

    f_a = geom.inertial2body(Q, A).x
    f_b = geom.inertial2body(Q, B).x

    if sign(f_a) == sign(f_b): # both are wrong - use the old way
        t = CurvePt.get_lerp_time_unclamped_3(A, B, Q)
        t = CurvePt.clamp(t, 0.0, 1.0)
        return t, VecSE2.lerp(A, B, t)

    iter = 1
    while iter <= max_iter:
        c = (a+b)/2 # new midpoint
        footpoint = VecSE2.lerp(A, B, c)
        f_c = geom.inertial2body(Q, footpoint).x
        if abs(f_c) < tolerance: # solution found
            return c, footpoint
        if sign(f_c) == sign(f_a):
            a, f_a = c, f_c
        else:
            b = c
        iter += 1

    # Maximum number of iterations passed
    # This will occur when we project with a point that is not actually in the range,
    # and we converge towards one edge

    if a == 0.0:
        return 0.0, A
    elif b == 1.0:
        return 1.0, B
    else:
        warnings.warn("get_closest_perpendicular_point_between_points - should not happen")
        c = (a+b)/2  # should not happen
        return c, VecSE2.lerp(A, B, c)


def proj_1(posG: VecSE2.VecSE2, lane: Lane, roadway: Roadway, move_along_curves: bool = True):
    curveproj = CurvePt.proj(posG, lane.curve)
    rettag = lane.tag
    if curveproj.ind == CurvePt.CurveIndex(0, 0.0) and has_prev(lane):
        pt_lo = prev_lane_point(lane, roadway)
        pt_hi = lane.curve[0]
        t = CurvePt.get_lerp_time_unclamped_2(pt_lo, pt_hi, posG)
        if t <= 0.0 and move_along_curves:
            return proj_1(posG, prev_lane(lane, roadway), roadway)
        elif t < 1.0:
            assert ((not move_along_curves) or 0.0 <= t < 1.0)
            # t was computed assuming a constant angle
            # this is not valid for the large distances and angle disparities between lanes
            # thus we now use a bisection search to find the appropriate location

            t, footpoint = get_closest_perpendicular_point_between_points(pt_lo.pos, pt_hi.pos, posG)

            ind = CurvePt.CurveIndex(-1, t)
            curveproj = CurvePt.get_curve_projection(posG, footpoint, ind)
    elif curveproj.ind == CurvePt.curveindex_end(lane.curve) and has_next(lane):
        pt_lo = lane.curve[-1]
        pt_hi = next_lane_point(lane, roadway)
        t = CurvePt.get_lerp_time_unclamped_2(pt_lo, pt_hi, posG)
        if t >= 1.0 and move_along_curves:
            return proj_1(posG, next_lane(lane, roadway), roadway)
        elif t >= 0.0:
            assert ((not move_along_curves) or 0.0 <= t < 1.0)
            # t was computed assuming a constant angle
            # this is not valid for the large distances and angle disparities between lanes
            # thus we now use a bisection search to find the appropriate location

            t, footpoint = get_closest_perpendicular_point_between_points(pt_lo.pos, pt_hi.pos, posG)

            ind = CurvePt.CurveIndex(len(lane.curve) - 1, t)
            curveproj = CurvePt.get_curve_projection(posG, footpoint, ind)
    #print(rettag.segment, rettag.lane)
    return RoadProjection(curveproj, rettag)


def proj_2(posG: VecSE2.VecSE2, roadway: Roadway):

    best_dist2 = math.inf
    best_proj = RoadProjection(CurvePt.CurveProjection(CurvePt.CurveIndex(-1, -1), None, None), NULL_LANETAG)

    for seg in roadway.segments:
        for lane in seg.lanes:
            roadproj = proj_1(posG, lane, roadway, move_along_curves=False)  # return RoadProjection
            targetlane = roadway.get_by_tag(roadproj.tag)  # return Lane
            footpoint = targetlane.get_by_ind_roadway(roadproj.curveproj.ind, roadway)
            vec = posG - footpoint.pos
            dist2 = VecE2.normsquared(VecE2.VecE2(vec.x, vec.y))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_proj = roadproj

    return best_proj





