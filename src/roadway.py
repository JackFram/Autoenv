# Zhihao Zhang
# NGSIM dataset processor roadway class

import re
import math
import os
from src.Vec import VecE2, VecSE2
from src.curves import CurvePt
from src.Roadway import roadway

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
            centerline.append(CurvePt.CurvePt(VecSE2.VecSE2(line[j], theta), s))
        s = centerline[npts - 2].s + (line[npts - 1] - line[npts - 2]).hypot()
        theta = (line[npts - 1] - line[npts - 2]).atan()
        centerline.append(CurvePt.CurvePt(VecSE2.VecSE2(line[npts - 1], theta), s))
        retval[name] = centerline
    return retval


def read_roadway(input_params: RoadwayInputParams):
    boundaries = read_boundaries(input_params.filepath_boundaries)
    centerlines = read_centerlines(input_params.filepath_centerlines)
    name = os.path.splitext(os.path.split('../data/centerlines80.txt')[1])[0]
    return NGSIMRoadway(name, boundaries, list(centerlines.values()))


def write_lwpolyline(pts: list, handle_int: int, is_closed: bool = False):
    N = len(pts)
    print("  0")
    print("LWPOLYLINE")
    print("  5") # handle (increases)
    print( "B{}".format(handle_int))
    print("100") # subclass marker
    print("AcDbEntity")
    print("  8") # layer name
    print("150")
    print("  6") # linetype name
    print("ByLayer")
    print(" 62") # color number
    print("  256")
    print("370") # lineweight enum
    print("   -1")
    print("100") # subclass marker
    print("AcDbPolyline")
    print(" 90") # number of vertices
    print("   {}".format(N))
    print(" 70") # 0 is default, 1 is closed
    print("    {}".format(1 if is_closed else 0))
    print(" 43") # 0 is constant width
    print("0")

    for i in range(N):
        print(" 10")
        print("{:.3f}".format(pts[i].pos.x))
        print(" 20")
        print("{:.3f}".format(pts[i].pos.y))


# function convert_curves_feet_to_meters!(roadway::Roadway)
#     for seg in roadway.segments
#         for lane in seg.lanes
#             for (i,P) in enumerate(lane.curve)
#                 lane.curve[i] = CurvePt(
#                         VecSE2(P.pos.x*METERS_PER_FOOT, P.pos.y*METERS_PER_FOOT, P.pos.θ),
#                         P.s*METERS_PER_FOOT, P.k/METERS_PER_FOOT, P.kd/METERS_PER_FOOT)
#             end
#         end
#     end
#     roadway
# end

# def write_dxf(roadway: NGSIMRoadway):
#     dirname, filename = os.path.split(os.path.abspath(__file__))
#     fp = open(os.path.join(dirname, "../data/template.dxf"), 'r')
#     lines = fp.readlines()
#     fp.close()
#     i = "ENTITIES\n" in lines
#     i != 0 || error("ENTITIES section not found")
# 
#     # write out header
#     for j in 1 : i
#         print(io, lines[j])
#     end
# 
#     # write out the lanes
#     for (handle_int, lane) in enumerate(roadway.centerlines)
#         write_lwpolyline(io, lane, handle_int)
#     end
# 
#     # write out tail
#     for j in i+1 : length(lines)
#         print(io, lines[j])
#     end
# end

def write_roadways_to_dxf():
    roadway_input_80 = RoadwayInputParams(os.path.join(DIR, "../data/boundaries80.txt"),
                                          os.path.join(DIR, "../data/centerlines80.txt"))
    roadway_input_101 = RoadwayInputParams(os.path.join(DIR, "../data/boundaries101.txt"),
                                           os.path.join(DIR, "../data/centerlines101.txt"))

    ngsimroadway_80 = read_roadway(roadway_input_80)
    ngsimroadway_101 = read_roadway(roadway_input_101)

    # open(io->write_dxf(io, ROADWAY_80), os.path.join(DIR, "../data/ngsim_80.dxf"), "w")
    # open(io->write_dxf(io, ROADWAY_101), os.path.join(DIR, "../data/ngsim_101.dxf"), "w")

# def write_roadways_from_dxf():
#
    # roadway_80 = open(io->read_dxf(io, Roadway, dist_threshold_lane_connect=2.0), os.path.join(DIR, "../data/ngsim_80.dxf"), "r")
    # roadway_101 = open(io->read_dxf(io, Roadway, dist_threshold_lane_connect=2.0), os.path.join(DIR, "../data/ngsim_101.dxf"), "r")

    # also converts to meters
    # convert_curves_feet_to_meters!(roadway_80)
    # convert_curves_feet_to_meters!(roadway_101)

    # open(io->write(io, roadway_80), os.path.join(DIR, "../data/ngsim_80.txt"), "w")
    # open(io->write(io, roadway_101), os.path.join(DIR, "../data/ngsim_101.txt"), "w")

ROADWAY_80 = roadway.read_roadway(open(os.path.join(DIR, "../data/ngsim_80.txt"), "r"))
ROADWAY_101 = roadway.read_roadway(open(os.path.join(DIR, "../data/ngsim_101.txt"), "r"))

