from src.Basic import Vehicle
from src.Record.frame import Frame, copyto
from src.Basic.Vehicle import read_def, read_state


class RecordFrame:
    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi

    def __len__(self):
        return self.hi - self.lo + 1

    def write(self, fp):
        fp.write("%d %d" % (self.lo, self.hi))


def read_frame(fp):
    tokens = fp.readline().strip().split(' ')
    lo = int(tokens[0])
    hi = int(tokens[1])
    return RecordFrame(lo, hi)


class RecordState:
    def __init__(self, state: Vehicle.VehicleState, id: int):
        self.state = state  # Dict
        self.id = id


class ListRecord:
    def __init__(self, timestep: float, frames: list, states: list, defs: dict):
        """
        timestep::Float64
        frames::Vector{RecordFrame}
        states::Vector{RecordState}
        defs::Dict{I, D}

        """
        self.timestep = timestep
        self.frames = frames
        self.states = states
        self.defs = defs

    def write(self, fp):
        fp.write("ListRecord{%s, %s, %s}(%d frames)\n" % ('NGSIM_TIMESTEP', 'Array{RecordFrame}', 'Array{RecordState{VehicleState, Int}}', len(self.frames)))
        fp.write("%.16e\n" % self.timestep)

        # defs
        fp.write(str(len(self.defs)))
        fp.write("\n")
        for id in self.defs:
            fp.write(str(id))
            fp.write("\n")
            self.defs[id].write(fp)
            fp.write("\n")

        # ids & states
        fp.write(str(len(self.states)))
        fp.write("\n")
        for recstate in self.states:
            fp.write(str(recstate.id))
            fp.write("\n")
            recstate.state.write(fp)
            fp.write("\n")

        # frames
        fp.write(str(len(self.frames)))
        fp.write("\n")
        for recframe in self.frames:
            recframe.write(fp)
            fp.write("\n")

    def n_objects_in_frame(self, frame_index: int):
        return len(self.frames[frame_index])

    @property
    def nframes(self):
        return len(self.frames)

    @property
    def nstates(self):
        return len(self.states)

    @property
    def nids(self):
        return len(self.defs.keys())


def read_trajdata(fp):
    lines = fp.readline()  # skip first line
    # lines = fp.readline()  # skip second line

    timestep = float(fp.readline())
    defs = dict()

    # read defs
    n = int(fp.readline())
    for i in range(n):
        id = int(fp.readline())
        # TODO: check if need parse /n
        defs[id] = read_def(fp)

    # read states
    n = int(fp.readline())
    states = [None for i in range(n)]
    for i in range(n):
        id = int(fp.readline())
        state = read_state(fp)
        states[i] = RecordState(state, id)

    # read frames
    n = int(fp.readline())
    frames = [None for i in range(n)]
    for i in range(n):
        frames[i] = read_frame(fp)

    return ListRecord(timestep, frames, states, defs)


class SceneRecord:
    def __init__(self):
        '''
        frames::List{Frame{Vehicle}}
        timestep::Float64
        nframes::Int # number of active Frames
        '''
        self.frames = []
        self.timestep = 0
        self.nframes = 0

    def __getitem__(self, item):
        return self.frames[0-item]

    def init(self, capacity: int, timestep: float, frame_capacity: int = 100):
        frames = []
        for i in range(capacity):
            frame = Frame()
            frame.init(frame_capacity)
            frames.append(frame)

        self.frames = frames
        self.timestep = timestep
        self.nframes = 0

    @property
    def capacity(self):
        return len(self.frames)

    def empty(self):
        self.nframes = 0

    def insert(self, frame: Frame, pastframe: int=0):
        self.frames[0 - pastframe] = copyto(self.frames[0 - pastframe], frame)

    def push_back_records(self):
        for i in range(min(self.nframes + 1, self.capacity) - 1, 0, -1):
            self.frames[i] = copyto(self.frames[i], self.frames[i - 1])

    def update(self, frame: Frame):
        self.push_back_records()
        self.insert(frame, 0)
        self.nframes = min(self.nframes + 1, self.capacity)


def frame_inbounds(rec: ListRecord, frame_index: int):
    return 0 <= frame_index < rec.nframes


def pastframe_inbounds(rec: SceneRecord, pastframe: int):
    return 0 <= 0 - pastframe <= rec.nframes - 1


def get_elapsed_time_3(rec: SceneRecord, pastframe_farthest_back: int, pastframe_most_recent: int):
    return (pastframe_most_recent - pastframe_farthest_back)*rec.timestep


def get_def(rec: ListRecord, id: int):
    return rec.defs[id]


def get_vehicle(rec: ListRecord, stateindex: int):
    recstate = rec.states[stateindex]
    # print(recstate.id)
    return Vehicle.Vehicle(recstate.state, get_def(rec, recstate.id), recstate.id)


def get_scene(frame: Frame, rec: ListRecord, frame_index: int):
    frame.empty()

    if frame_inbounds(rec, frame_index):
        recframe = rec.frames[frame_index]
        print(recframe.lo, recframe.hi)
        for stateindex in range(recframe.lo, recframe.hi + 1):
            frame.push(get_vehicle(rec, stateindex))

    return frame







