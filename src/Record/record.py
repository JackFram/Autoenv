from Basic import Vehicle


class RecordFrame:
    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi

    def __len__(self):
        return self.hi - self.lo + 1

    def write(self, fp):
        fp.write("%d %d" % (self.lo, self.hi))


class RecordState:
    def __init__(self, state: Vehicle.VehicleState, id: list):
        self.state = state  # Dict
        self.id = id  # Array


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
        fp.write("ListRecord{%s, %s, %s}(%d frames)" % ('NGSIM_TIMESTEP', 'Array{RecordFrame}', 'Array{RecordState{VehicleState, Int}}\n', len(self.frames)))
        fp.write("%.16e\n" % self.timestep)

        # defs
        fp.write(str(len(self.defs)))
        for id in self.defs:
            fp.write(str(id))
            fp.write("\n")
            self.defs[id].write(fp)
            fp.write("\n")

        # ids & states
        fp.write(str(len(self.states)))
        for recstate in self.states:
            fp.write(str(recstate.id))
            fp.write("\n")
            recstate.state.write(fp)
            fp.write("\n")

        # frames
        fp.write(str(len(self.frames)))
        for recframe in self.frames:
            recframe.write(fp)





