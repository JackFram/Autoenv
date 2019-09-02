# Collision Detection routines based on Minkowski techniques.



######################################

function cyclic_shift_left!(arr::AbstractVector, d::Int, n::Int=length(a))
    #=
    Perform a cyclic rotation of the elements in the array, in place
        d = amount to shift
        n = length of array (if we only want to work with first n elements)
    =#

    for i in 1 : gcd(d, n)
    # move i-th values of blocks

        temp = arr[i]
        j = i
        while true
            k = j + d
            if k > n
                k = k - n
            end
            if k == i
                break
            end
            arr[j] = arr[k]
            j = k
        end
        arr[j] = temp
    end

    arr
end

######################################

function get_signed_area(pts::Vector{VecE2}, npts::Int = length(pts))

    # https://en.wikipedia.org/wiki/Shoelace_formula
    # sign of -1 means clockwise, sign of 1 means counterclockwise

    retval = pts[npts].x*pts[1].y - pts[1].x*pts[npts].y
    for i in 1 : npts-1
        retval += pts[i].x * pts[i+1].y
        retval -= pts[i+1].x * pts[i].y
    end

    retval / 2
end
function get_edge(pts::Vector{VecE2}, i::Int, npts::Int=length(pts))
    a = pts[i]
    b = i+1 ≤ npts ? pts[i+1] : pts[1]
    LineSegment(a,b)
end

######################################

mutable struct ConvexPolygon
    pts::Vector{VecE2} # ordered counterclockwise along polygon boundary s.t. first edge has minimum polar angle in [0,2π]
    npts::Int # number of pts that are currently used (since we preallocate a longer array)
end

ConvexPolygon(npts::Int) = ConvexPolygon(Array{VecE2}(undef, npts), 0)
ConvexPolygon(pts::Vector{VecE2}) = ConvexPolygon(pts, length(pts))

function Base.iterate(poly::ConvexPolygon, i::Int=1)
    if i > length(poly)
        return nothing 
    end
    return (poly.pts[i], i+1)
end

Base.length(poly::ConvexPolygon) = poly.npts
Base.isempty(poly::ConvexPolygon) = poly.npts == 0
get_edge(P::ConvexPolygon, i::Int) = get_edge(P.pts, i, P.npts)
get_signed_area(poly::ConvexPolygon) = get_signed_area(poly.pts, poly.npts)
function Base.empty!(poly::ConvexPolygon)
    poly.npts = 0
    poly
end
function Base.copyto!(dest::ConvexPolygon, src::ConvexPolygon)
    dest.npts = src.npts
    copyto!(dest.pts, 1, src.pts, 1, src.npts)
    dest
end
function Base.push!(poly::ConvexPolygon, v::VecE2)
    poly.pts[poly.npts+1] = v
    poly.npts += 1
    poly
end
function Base.in(v::VecE2, poly::ConvexPolygon)

    # does NOT include pts on the physical boundary

    previous_side = 0

    for i in 1 : length(poly)
        seg = get_edge(poly, i)
        affine_segment = seg.B - seg.A
        affine_point = v - seg.A
        current_side = round(Int, sign(cross(affine_segment, affine_point))) # sign indicates side
        if current_side == 0
            # outside or over an edge
            return false
        elseif previous_side == 0
            # first segment
            previous_side = current_side
        elseif previous_side != current_side
            # all most be on the same side
            return false
        end
    end
    true
end
Base.in(v::VecSE2{Float64}, poly::ConvexPolygon) = in(convert(VecE2, v), poly)
function Base.show(io::IO, poly::ConvexPolygon)
    @printf(io, "ConvexPolygon: len %d (max %d pts)\n", length(poly), length(poly.pts))
    for i in 1 : length(poly)
        print(io, "\t"); show(io, poly.pts[i])
        print(io, "\n")
    end
end

AutomotiveDrivingModels.get_center(poly::ConvexPolygon) = sum(poly.pts) / poly.npts
function Vec.get_distance(poly::ConvexPolygon, v::VecE2; solid::Bool=true)
    if solid && in(v, poly)
        0.0
    else
        min_dist = Inf
        for i in 1 : length(poly)
            seg = get_edge(poly, i)
            min_dist = min(min_dist, get_distance(seg, v))
        end
        min_dist
    end
end

function ensure_pts_sorted_by_min_polar_angle!(poly::ConvexPolygon, npts::Int=poly.npts)

    @assert(npts ≥ 3)
    @assert(sign(get_signed_area(poly)) == 1) # must be counter-clockwise

    # ensure that edges are sorted by minimum polar angle in [0,2π]

    angle_start = Inf
    index_start = -1
    for i in 1 : npts
        seg = get_edge(poly.pts, i, npts)

        θ = atan(seg.B.y - seg.A.y, seg.B.x - seg.A.x)
        if θ < 0.0
            θ += 2π
        end

        if θ < angle_start
            angle_start = θ
            index_start = i
        end
    end

    if index_start != 1
        cyclic_shift_left!(poly.pts, index_start-1, npts)
    end
    poly
end
function shift!(poly::ConvexPolygon, v::VecE2)
    for i in 1 : length(poly)
        poly.pts[i] += v
    end
    ensure_pts_sorted_by_min_polar_angle!(poly)
    poly
end
function rotate!(poly::ConvexPolygon, θ::Float64)
    for i in 1 : length(poly)
        poly.pts[i] = Vec.rot(poly.pts[i], θ)
    end
    ensure_pts_sorted_by_min_polar_angle!(poly)
    poly
end
function mirror!(poly::ConvexPolygon)
    for i in 1 : length(poly)
        poly.pts[i] = -poly.pts[i]
    end
    ensure_pts_sorted_by_min_polar_angle!(poly)
    poly
end

function minkowksi_sum!(retval::ConvexPolygon, P::ConvexPolygon, Q::ConvexPolygon)

    #=
    For two convex polygons P and Q in the plane with m and n vertices, their Minkowski sum is a
    convex polygon with at most m + n vertices and may be computed in time O (m + n) by a very simple procedure,
    which may be informally described as follows.

    Assume that the edges of a polygon are given and the direction, say, counterclockwise, along the polygon boundary.
    Then it is easily seen that these edges of the convex polygon are ordered by polar angle.
    Let us merge the ordered sequences of the directed edges from P and Q into a single ordered sequence S.
    Imagine that these edges are solid arrows which can be moved freely while keeping them parallel to their original direction.
    Assemble these arrows in the order of the sequence S by attaching the tail of the next arrow to the head of the previous arrow.
    It turns out that the resulting polygonal chain will in fact be a convex polygon which is the Minkowski sum of P and Q.
    =#

    empty!(retval)

    index_P = 1
    index_Q = 1
    θp = get_polar_angle(get_edge(P, index_P))
    θq = get_polar_angle(get_edge(Q, index_Q))

    while index_P ≤ length(P) || index_Q ≤ length(Q)
        # select next edge with minimum polar angle

        if θp == θq
            seg_p = get_edge(P, index_P)
            seg_q = get_edge(Q, index_Q)
            O = isempty(retval) ? P.pts[1] + Q.pts[1] : retval.pts[retval.npts]
            push!(retval, O + seg_p.B - seg_p.A + seg_q.B - seg_q.A)
            index_P += 1
            θp = index_P ≤ length(P) ? get_polar_angle(get_edge(P, index_P)) : Inf
            index_Q += 1
            θq = index_Q ≤ length(Q) ? get_polar_angle(get_edge(Q, index_Q)) : Inf
        elseif θp ≤ θq
            seg = get_edge(P, index_P)
            O = isempty(retval) ? P.pts[1] + Q.pts[1] : retval.pts[retval.npts]
            push!(retval, O + seg.B - seg.A)
            index_P += 1
            θp = index_P ≤ length(P) ? get_polar_angle(get_edge(P, index_P)) : Inf
        else
            seg = get_edge(Q, index_Q)
            O = isempty(retval) ? P.pts[1] + Q.pts[1] : retval.pts[retval.npts]
            push!(retval, O + seg.B - seg.A)
            index_Q += 1
            θq = index_Q ≤ length(Q) ? get_polar_angle(get_edge(Q, index_Q)) : Inf
        end
    end

    ensure_pts_sorted_by_min_polar_angle!(retval)

    retval
end
function minkowski_difference!(retval::ConvexPolygon, P::ConvexPolygon, Q::ConvexPolygon)

    #=
    The minkowski difference is what you get by taking the minkowski sum of a shape and the mirror of another shape.
    So, your second shape gets flipped about the origin (all of its points are negated).

    The idea is that you do a binary operation on two shapes to get a new shape,
    and if the origin (the zero vector) is inside that shape, then they are colliding.
    =#

    minkowksi_sum!(retval, P, mirror!(Q))
end

function is_colliding(P::ConvexPolygon, Q::ConvexPolygon, temp::ConvexPolygon=ConvexPolygon(length(P) + length(Q)))
    minkowski_difference!(temp, P, Q)
    in(VecE2(0,0), temp)
end
function Vec.get_distance(P::ConvexPolygon, Q::ConvexPolygon, temp::ConvexPolygon=ConvexPolygon(length(P) + length(Q)))
    minkowski_difference!(temp, P, Q)
    get_distance(VecE2(0,0), temp)
end

function to_oriented_bounding_box!(retval::ConvexPolygon, center::VecSE2{Float64}, len::Float64, wid::Float64)

    @assert(len > 0)
    @assert(wid > 0)
    @assert(!isnan(center.θ))
    @assert(!isnan(center.x))
    @assert(!isnan(center.y))

    x = polar(len/2, center.θ)
    y = polar(wid/2, center.θ+π/2)

    C = convert(VecE2,center)
    retval.pts[1] =  x - y + C
    retval.pts[2] =  x + y + C
    retval.pts[3] = -x + y + C
    retval.pts[4] = -x - y + C
    retval.npts = 4

    AutomotiveDrivingModels.ensure_pts_sorted_by_min_polar_angle!(retval)

    retval
end
get_oriented_bounding_box(center::VecSE2{Float64}, len::Float64, wid::Float64) = to_oriented_bounding_box!(ConvexPolygon(4), center, len, wid)
function to_oriented_bounding_box!(retval::ConvexPolygon, veh::Vehicle, center::VecSE2{Float64} = get_center(veh))

    # get an oriented bounding box at the vehicle's position

    to_oriented_bounding_box!(retval, center, veh.def.length, veh.def.width)

    retval
end
get_oriented_bounding_box(veh::Vehicle, center::VecSE2{Float64} = get_center(veh)) = to_oriented_bounding_box!(ConvexPolygon(4), veh, center)

######################################

function is_colliding(ray::VecSE2{Float64}, poly::ConvexPolygon)
    # collides if at least one of the segments collides with the ray

    for i in 1 : length(poly)
        seg = get_edge(poly, i)
        if intersects(ray, seg)
            return true
        end
    end
    false
end
function get_collision_time(ray::VecSE2{Float64}, poly::ConvexPolygon, ray_speed::Float64)
    min_col_time = Inf
    for i in 1 : length(poly)
        seg = get_edge(poly, i)
        col_time = get_intersection_time(Projectile(ray, ray_speed), seg)
        if !isnan(col_time) && col_time < min_col_time
            min_col_time = col_time
        end
    end
    min_col_time
end

######################################


struct CPAMemory
    vehA::ConvexPolygon # bounding box for vehicle A
    vehB::ConvexPolygon # bounding box for vehicle B
    mink::ConvexPolygon # minkowski bounding box

    CPAMemory() = new(ConvexPolygon(4), ConvexPolygon(4), ConvexPolygon(8))
end
is_colliding(mem::CPAMemory) = is_colliding(mem.vehA, mem.vehB, mem.mink)
Vec.get_distance(mem::CPAMemory) = get_distance(mem.vehA, mem.vehB, mem.mink)
function get_time_and_dist_of_closest_approach(a::Vehicle, b::Vehicle, mem::CPAMemory=CPAMemory())

    to_oriented_bounding_box!(mem.vehA, a)
    to_oriented_bounding_box!(mem.vehB, b)
    minkowksi_sum!(mem.mink, mem.vehA, mem.vehB)

    rel_pos = convert(VecE2, b.state.posG) - a.state.posG
    rel_velocity = polar(b.state.v, b.state.posG.θ) - polar(a.state.v, a.state.posG.θ)
    ray_speed = norm(VecE2(rel_velocity))
    ray = VecSE2(rel_pos, atan(rel_velocity))

    if in(convert(VecE2, ray), mem.mink)
        return (0.0, 0.0)
    end

    best_t_CPA = NaN
    best_d_CPA = Inf
    if_no_col_skip_eval = false

    for i in 1 : length(mem.mink)
        seg = get_edge(mem.mink, i)
        t_CPA, d_CPA = get_time_and_dist_of_closest_approach(ray, seg, ray_speed, if_no_col_skip_eval)
        if d_CPA < best_d_CPA
            best_t_CPA = t_CPA
            best_d_CPA = d_CPA
            if d_CPA == 0.0
                if_no_col_skip_eval = true
            end
        end
    end

    (best_t_CPA, best_d_CPA)
end

_bounding_radius(veh::Vehicle) = sqrt(veh.def.length*veh.def.length/4 + veh.def.width*veh.def.width/4)

"""
A fast collision check to remove things clearly not colliding
"""
function is_potentially_colliding(A::Vehicle, B::Vehicle)
    Δ² = normsquared(VecE2(A.state.posG - B.state.posG))
    r_a = _bounding_radius(A)
    r_b = _bounding_radius(B)
    Δ² ≤ r_a*r_a + 2*r_a*r_b + r_b*r_b
end

function is_colliding(A::Vehicle, B::Vehicle, mem::CPAMemory=CPAMemory())
    if is_potentially_colliding(A, B)
        to_oriented_bounding_box!(mem.vehA, A)
        to_oriented_bounding_box!(mem.vehB, B)
        return is_colliding(mem)
    end
    false
end
function Vec.get_distance(A::Vehicle, B::Vehicle, mem::CPAMemory=CPAMemory())
    to_oriented_bounding_box!(mem.vehA, A)
    to_oriented_bounding_box!(mem.vehB, B)
    get_distance(mem)
end

struct CollisionCheckResult
    is_colliding::Bool
    A::Int # index of 1st vehicle
    B::Int # index of 2nd vehicle
    # TODO: penetration vector?
end

"""
Loops through the scene and finds the first collision between a vehicle and scene[target_index]
"""
function get_first_collision(scene::EntityFrame{S,D,I}, target_index::Int, mem::CPAMemory=CPAMemory()) where {S<:VehicleState,D<:Union{VehicleDef, BicycleModel},I}
    A = target_index
    vehA = scene[A]
    vehA = convert(Vehicle,vehA)
    to_oriented_bounding_box!(mem.vehA, vehA)
    for (B,vehB) in enumerate(scene)
        vehB = convert(Vehicle,vehB)
        if B != A
            to_oriented_bounding_box!(mem.vehB, vehB)
            if is_potentially_colliding(vehA, vehB) && is_colliding(mem)
                return CollisionCheckResult(true, A, B)
            end
        end
    end

    CollisionCheckResult(false, A, 0)
end

"""
Loops through the scene and finds the first collision between any two vehicles
"""
function get_first_collision(scene::EntityFrame{S,D,I}, vehicle_indeces::AbstractVector{Int}, mem::CPAMemory=CPAMemory()) where {S<:VehicleState,D<:Union{VehicleDef, BicycleModel},I}

    N = length(vehicle_indeces)
    for (a,A) in enumerate(vehicle_indeces)
        vehA = scene[A]
        vehA = convert(Vehicle,vehA)
        to_oriented_bounding_box!(mem.vehA, vehA)
        for b in a +1 : length(vehicle_indeces)
            B = vehicle_indeces[b]
            vehB = scene[B]
            vehB = convert(Vehicle,vehB)
            if is_potentially_colliding(vehA, vehB)
                to_oriented_bounding_box!(mem.vehB, vehB)
                if is_colliding(mem)
                    return CollisionCheckResult(true, A, B)
                end
            end
        end
    end

    CollisionCheckResult(false, 0, 0)
end
get_first_collision(scene::EntityFrame{S,D,I}, mem::CPAMemory=CPAMemory()) where {S<:VehicleState,D<:Union{VehicleDef, BicycleModel},I} = get_first_collision(scene, 1:length(scene), mem)
is_collision_free(scene::EntityFrame{S,D,I}, mem::CPAMemory=CPAMemory()) where {S<:VehicleState,D<:Union{VehicleDef, BicycleModel},I} = !(get_first_collision(scene, mem).is_colliding)
is_collision_free(scene::EntityFrame{S,D,I}, vehicle_indeces::AbstractVector{Int}, mem::CPAMemory=CPAMemory()) where {S<:VehicleState,D<:Union{VehicleDef, BicycleModel},I} = get_first_collision(scene, vehicle_indeces, mem).is_colliding

###

"""
    CollisionCallback

Terminates the simulation once a collision occurs
"""
@with_kw struct CollisionCallback
    mem::CPAMemory=CPAMemory()
end
function run_callback(
    callback::CollisionCallback,
    rec::EntityQueueRecord{S,D,I},
    roadway::R,
    models::Dict{I,M},
    tick::Int,
    ) where {S,D,I,R,M<:DriverModel}

    return !is_collision_free(rec[0], callback.mem)
end
