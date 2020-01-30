using TrajectoryOptimization
using Statistics
using Random
using StaticArrays
using LinearAlgebra
using MeshCat
using TrajOptPlots
using DataFrames
const TO = TrajectoryOptimization
include("plotting.jl")

model = Dynamics.Quadrotor2{UnitQuaternion{Float64,CayleyMap}}()
if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

max_con_viol = 1.0e-8
T = Float64
verbose = true

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    cost_tolerance=1e-4,
    iterations=300)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)


function gen_quad_zigzag(Rot; use_rot=Rot<:UnitQuaternion, costfun=:Quadratic)
    model = Dynamics.Quadrotor2{Rot}(use_rot=use_rot)
    n,m = size(model)

    # discretization
    N = 101 # number of knot points
    tf = 5.0
    dt = tf/(N-1) # total time

    # Initial condition
    x0_pos = @SVector [0., -10., 1.]
    x0 = Dynamics.build_state(model, x0_pos, I(UnitQuaternion), zeros(3), zeros(3))

    # cost
    costfun == :QuatLQR ? sq = 0 : sq = 1
    Q_diag = Dynamics.fill_state(model, 1e-5, 1e-5*sq, 1e-3, 1e-3)
    R = Diagonal(@SVector fill(1e-4,m))
    q_nom = I(UnitQuaternion)
    v_nom, ω_nom = zeros(3), zeros(3)
    x_nom = Dynamics.build_state(model, zeros(3), q_nom, v_nom, ω_nom)
    if costfun == :QuatLQR
        cost_nom = QuatLQRCost(Diagonal(Q_diag), R, x_nom, w=0.0)
    else
        cost_nom = LQRCost(Diagonal(Q_diag), R, x_nom)
    end

    # waypoints
    wpts = [(@SVector [10,0,1.]),
            (@SVector [-10,0,1.]),
            (@SVector [0,10,1.])]
    times = [33, 66, 101]
    Qw_diag = Dynamics.fill_state(model, 1e3,1*sq,1,1)
    Qf_diag = Dynamics.fill_state(model, 10., 100*sq, 10, 10)
    xf = Dynamics.build_state(model, wpts[end], I(UnitQuaternion), zeros(3), zeros(3))

    costs = map(1:length(wpts)) do i
        r = wpts[i]
        xg = Dynamics.build_state(model, r, q_nom, v_nom, ω_nom)
        if times[i] == N
            Q = Diagonal(Qf_diag)
            w = 1.0
        else
            Q = Diagonal(1e-3*Qw_diag)
            w = 0.0
        end
        if costfun == :QuatLQR
            QuatLQRCost(Q, R, xg, w=w)
        else
            LQRCost(Q, R, xg)
        end
    end

    costs_all = map(1:N) do k
        i = findfirst(x->(x ≥ k), times)
        if k ∈ times
            costs[i]
        else
            cost_nom
        end
    end

    obj = Objective(costs_all)

    # Initialization
    u0 = @SVector fill(0.5*9.81/4, m)
    U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

    # Problem
    prob = Problem(model, obj, xf, tf, x0=x0)
    solver = iLQRSolver(prob, opts_ilqr)
    initial_controls!(solver, U_hover)
    solver.opts.verbose = true

    return solver
end
solver = gen_quad_zigzag(UnitQuaternion{Float64,CayleyMap}, costfun=:QuatLQR)
solver = gen_quad_zigzag(UnitQuaternion{Float64,MRPMap}, costfun=:QuatLQR)
solver = gen_quad_zigzag(UnitQuaternion{Float64,ExponentialMap}, costfun=:QuatLQR)
solver = gen_quad_zigzag(UnitQuaternion{Float64,VectorPart}, costfun=:QuatLQR)
solver = gen_quad_zigzag(MRP{Float64}, use_rot=false)
solver = gen_quad_zigzag(RodriguesParam{Float64}, use_rot=false)
solver = gen_quad_zigzag(RPY{Float64})

solve!(solver)
visualize!(vis, solver.model, solver.Z)

function log_solve!(solver, data; kwargs...)
    b = benchmark_solve!(solver, data; kwargs...)
    push!(data[:costfun], cost_type(solver))
    push!(data[:rotation], rot_type(solver))
end

function run_all(;kwargs...)
    # Initial data structure
    data = Dict{Symbol,Vector}(:costfun=>Symbol[], :iterations=>Int[], :rotation=>Symbol[],
        :time=>Float64[], :cost=>Float64[])

    # Baseline methods
    println("Baseline methods")
    log_solve!(gen_quad_zigzag(UnitQuaternion{Float64, IdentityMap}, use_rot=false), data; kwargs...)
    log_solve!(gen_quad_zigzag(RodriguesParam{Float64}, use_rot=false), data; kwargs...)
    log_solve!(gen_quad_zigzag(MRP{Float64}, use_rot=false), data; kwargs...)
    log_solve!(gen_quad_zigzag(RPY{Float64}, use_rot=false), data; kwargs...)

    # Quaternion Methods
    println("Quaternion methods")
    for rmap in [ExponentialMap, CayleyMap, MRPMap, VectorPart]
        for costfun in [:Quadratic, :QuatLQR]
            log_solve!(gen_quad_zigzag(UnitQuaternion{Float64, rmap}, costfun=costfun), data; kwargs...)
        end
    end
    return data
end

data = run_all(samples=1, evals=1)

df = DataFrame(data)
df.rots = string.(short_names.(df.rotation))
df.time_per_iter = df.time ./ df.iterations
quats = in([:ExponentialMap, :CayleyMap, :MRPMap, :VectorPart])
base = in([:IdentityMap, :MRP, :RP, :RPY])

df_ = df[base.(df.rotation),:]
bar_comparison(df_.rots, df_.time)


############################################################################################
# PLOT 1: Time per iteration
############################################################################################

tpi = df[base.(df.rotation),:]
tpi = by(df, :rots, :time_per_iter=>minimum)
coord = Coordinates(tpi.rots, tpi.time_per_iter_minimum)

p = @pgf Axis(
    {
        ybar,
        ylabel="time per iteration (ms)",
        enlargelimits = 0.15,
        x="0.9cm",
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=tpi.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coord),
    # Legend(["90 degrees", "180 degrees", "270 degrees"])
)

pgfsave("figs/zig_time_per_iter.tikz", p, include_preamble=false)


############################################################################################
# PLOT 1: Time per iteration
############################################################################################

base = df[base.(df.rotation),:]
quat = df[quats.(df.rotation),:]
bar_comparison(base.rots, base.iterations)
qlqr = quat[quat.costfun .== :QuatLQR,:]
quad = quat[quat.costfun .== :Quadratic,:]
coord_base = Coordinates(base.rots, base.iterations)
coord1 = Coordinates(qlqr.rots, qlqr.iterations)
coord2 = Coordinates(quad.rots, quad.iterations)

p = @pgf Axis(
    {
        ybar,
        ylabel="iterations",
        "enlarge x limits" = 0.20,
        x="1.2cm",
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=tpi.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coord1),
    Plot(coord2),
    Legend(["Geodesic", "Error Quadratic"])
)
pgfsave("figs/zig_quat_iters.tikz", p, include_preamble=false)

p = @pgf Axis(
    {
        ybar,
        ylabel="iterations",
        "enlarge x limits" = 0.20,
        x="1.2cm",
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=tpi.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coord_base),
    # Legend(["Geodesic", "Error Quadratic"])
)
pgfsave("figs/zig_base_iters.tikz", p, include_preamble=false)
