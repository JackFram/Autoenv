from src.Vec import VecSE2, VecE2
from src.Roadway import roadway, utils
from src.curves import CurvePt
import math

"""
Frenet
______
roadind: road index
s: distance along lane
t: lane offset, positive is to left. zero point is the centerline of the lane.
ϕ: lane relative heading
"""


class Frenet:
    def __init__(self, roadind: roadway.RoadIndex = None, s: float = None, t: float = None, phi: float = None):
        self.roadind = roadind
        self.s = s
        self.t = t
        self.phi = phi

    def set(self, posG: VecSE2.VecSE2, roadWay: roadway.Roadway):
        # roadind: roadway.RoadIndex, s: float, t: float, phi: float
        roadproj = roadway.proj_2(posG, roadWay)
        roadind = roadway.RoadIndex(roadproj.curveproj.ind, roadproj.tag)
        s = roadWay.get_by_roadindex(roadind).s
        t = roadproj.curveproj.t
        phi = utils.mod2pi2(roadproj.curveproj.phi)
        self.roadind = roadind
        self.s = s
        self.t = t
        self.phi = phi


class AgentClass:
    MOTORCYCLE = 1
    CAR = 2
    TRUCK = 3
    PEDESTRIAN = 4


"""
    Vehicle definition which contains a class and a bounding box.
"""


class VehicleDef:
    def __init__(self, class_: int = AgentClass.CAR, length_: int = 4.0, width_: int = 1.8):
        self.class_ = class_
        self.length_ = length_
        self.width_ = width_

    def write(self, fp):
        fp.write("%d %.16e %.16e" % (self.class_, self.length_, self.width_))


def read_def(fp):
    tokens = fp.readline().strip().split(' ')
    class_ = int(tokens[0])
    length_ = int(tokens[1])
    width_ = int(tokens[2])
    return VehicleDef(class_, length_, width_)


NULL_VEHICLEDEF = VehicleDef(AgentClass.CAR, None, None)


class VehicleState:
    def __init__(self, posG: VecSE2.VecSE2 = None, posF: Frenet = None, v: float = None):
        self.posG = posG
        self.posF = posF
        self.v = v

    def set(self, posG: VecSE2.VecSE2, roadWay: roadway.Roadway, v: float):
        self.posG = posG
        self.posF = Frenet()
        self.posF.set(posG, roadWay)
        self.v = v

    def write(self, fp):
        fp.write("%.16e %.16e %.16e" % (self.posG.x, self.posG.y, self.posG.theta))
        fp.write(" %d %.16e %d %d" % (self.posF.roadind.ind.i, self.posF.roadind.ind.t,
                                      self.posF.roadind.tag.segment, self.posF.roadind.tag.lane))
        fp.write(" %.16e %.16e %.16e" % (self.posF.s, self.posF.t, self.posF.phi))
        fp.write(" %.16e" % self.v)


def read_state(fp):
    tokens = fp.readline().strip().split(' ')
    i = 0
    x = float(tokens[i])
    i += 1
    y = float(tokens[i])
    i += 1
    theta = float(tokens[i])
    i += 1
    posG = VecSE2.VecSE2(x, y, theta)
    ind_i = int(tokens[i])
    i += 1
    ind_t = float(tokens[i])
    i += 1
    tag_segment = int(tokens[i])
    i += 1
    tag_lane = int(tokens[i])
    i += 1
    roadind = roadway.RoadIndex(CurvePt.CurveIndex(ind_i, ind_t), roadway.LaneTag(tag_segment, tag_lane))
    s = float(tokens[i])
    i += 1
    t = float(tokens[i])
    i += 1
    phi = float(tokens[i])
    i += 1
    posF = Frenet(roadind, s, t, phi)
    v = float(tokens[i])
    return VehicleState(posG, posF, v)




class Vehicle:
    def __init__(self, state_: VehicleState, def_: VehicleDef, id: int):
        self.state = state_
        self.definition = def_
        self.id = id

    @property
    def get_center(self):
        return self.state.posG

    @property
    def get_footpoint(self):
        return self.state.posG + VecE2.polar(self.state.posF.t, self.state.posG.theta-self.state.posF.phi-math.pi/2)

    def get_targetpoint_delta(self, part: str):
        if part == "Front":
            return self.definition.length_/2*math.cos(self.state.posF.phi)
        elif part == "Center":
            return 0.0
        elif part == "Rear":
            return -self.definition.length_/2*math.cos(self.state.posF.phi)
        else:
            raise ValueError("Invalid TargetPoint!")


def get_lane_width(veh: Vehicle, roadway_: roadway.Roadway):
    lane = roadway_.get_by_tag(veh.state.posF.roadind.tag)

    if roadway.n_lanes_left(lane, roadway_) > 0:
        footpoint = veh.get_footpoint
        lane_left = roadway_.get_by_tag(roadway.LaneTag(lane.tag.segment, lane.tag.lane + 1))
        return -roadway.proj_1(footpoint, lane_left, roadway_).curveproj.t
    else:
        return lane.width


def get_markerdist_left(veh: Vehicle, roadway_: roadway.Roadway):
    t = veh.state.posF.t
    lane_width = get_lane_width(veh, roadway_)
    return lane_width / 2 - t


def get_markerdist_right(veh: Vehicle, roadway_: roadway.Roadway):
    t = veh.state.posF.t
    lane_width = get_lane_width(veh, roadway_)
    return lane_width / 2 + t


