
using TrajectoryOptimization
using Random
using StaticArrays
using LinearAlgebra
using MeshCat
using TrajOptPlots
using Rotations
import TrajectoryOptimization: Rotation
const TO = TrajectoryOptimization

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

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-8,
    resolve_feasible_problem=false,
    projected_newton=false,
    projected_newton_tolerance=1.0e-3)


model = Dynamics.Quadrotor2{UnitQuaternion{Float64,VectorPart}}()
n,m = 13,4

if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

function gen_quad_flip(Rot=UnitQuaternion{Float64,CayleyMap}; use_rot=Rot<:UnitQuaternion,
        costfun=:auto)
    model = Dynamics.Quadrotor2{Rot}(use_rot=use_rot)
    n,m = size(model)
    rsize = n-9

    # discretization
    N = 101 # number of knot points
    tf = 5.0
    dt = tf/(N-1) # total time

    # Initial condition
    x0_pos = @SVector [0., -1., 1.]
    x0 = Dynamics.build_state(model, x0_pos, I(UnitQuaternion), zeros(3), zeros(3))

    # cost
    if costfun == :auto
        if Rot <: UnitQuaternion{T,IdentityMap} where T || Rot <: RPY
            costfun = :Quadratic
        else
            costfun = :QuatLQR
        end
    end
    @show costfun
    costfun == :QuatLQR ? sq = 0 : sq = 1
    Q_diag = Dynamics.build_state(model, [1e-2,1e-2,5e-2], fill(1e-5,3)*sq, fill(1e-3,3), fill(1e-2,3))
    R = Diagonal(@SVector fill(1e-3,m))
    q_nom = I(UnitQuaternion)
    v_nom, ω_nom = zeros(3), zeros(3)
    x_nom = Dynamics.build_state(model, zeros(3), q_nom, v_nom, ω_nom)

    # waypoints
    ex = @SVector [1.0, 0, 0]
    wpts = [((@SVector [0, 0.5, 1.5,]),   expm(ex*deg2rad(90))),
            # ((@SVector [0, 0.2, 1.5]),    expm(ex*deg2rad(90))),
            # ((@SVector [0, 0.0, 2.0]),    expm(ex*deg2rad(135))),
            ((@SVector [0, 0.0, 2.5]),    expm(ex*deg2rad(180))),
            # ((@SVector [0, 0.0, 2.0]),    expm(ex*deg2rad(225))),
            ((@SVector [0,-0.5, 1.5]),    expm(ex*deg2rad(-90))),
            ((@SVector [0, 0.65, 1.0]),    expm(ex*deg2rad(360))),
            ((@SVector [0, 1.0, 1.0]),    expm(ex*deg2rad(360))),
            ]
    # times = [35, 41, 47, 51, 55, 61, 70, 101]
    times = [45, 51, 55, 75, 101]
    Qw_diag = Dynamics.build_state(model, [1e3,1e1,1e3], sq*(@SVector fill(5e3,rsize)), fill(1,3), fill(10,3))
    Qf_diag = Dynamics.fill_state(model, 10., 100*sq, 10, 10)
    xf = Dynamics.build_state(model, wpts[end][1], wpts[end][2], zeros(3), zeros(3))

    if costfun == :QuatLQR
        cost_nom = QuatLQRCost(Diagonal(Q_diag), R, xf, w=0.02)
    else
        cost_nom = LQRCost(Diagonal(Q_diag), R, xf)
    end

    costs = map(1:length(wpts)) do i
        r,q = wpts[i]
        xg = Dynamics.build_state(model, r, q, v_nom, [2pi/3, 0, 0])
        if times[i] == N
            Q = Diagonal(Qf_diag)
            w = 10.
        else
            Q = Diagonal(1e-3*Qw_diag)
            w = 100.
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

    # Constraints
    conSet = ConstraintSet(n,m,N)
    add_constraint!(conSet, GoalConstraint(xf,@SVector [1,2,3]), N:N)
    xmin = fill(-Inf,n)
    xmin[3] = 0.0
    bnd = BoundConstraint(n,m, x_min=xmin)
    add_constraint!(conSet, bnd, 1:N-1)

    # Initialization
    u0 = @SVector fill(0.5*9.81/4, m)
    tz = @SVector ones(4)
    tx = @SVector [-1,0,1,0.]
    ty = @SVector [0,-1,0, 1.]
    U_hover = [copy(u0) + ty*0e-3 - (k ∈ 40:41) * ty*0e-1 for k = 1:N-1] # initial hovering control trajectory
    # U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

    # Problem
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
    initial_controls!(prob, U_hover)
    prob

    # Infeasible start trajectory
    Dynamics.build_state(model, zeros(3), UnitQuaternion(I), zeros(3), zeros(3))
    model = prob.model
    X_guess = map(1:prob.N) do k
        t = (k-1)/prob.N
        x = (1-t)*x0 + t*xf
        Dynamics.build_state(model, position(model, x),
            expm(2pi*t*@SVector [1.,0,0]),
            Dynamics.linear_velocity(model, x),
            Dynamics.angular_velocity(model, x))
    end
    Z = deepcopy(prob.Z)
    TO.set_states!(Z, X_guess)
    # visualize!(vis, model, Z)
    initial_states!(prob, X_guess)

    opts_ilqr = iLQRSolverOptions{T}(verbose=false,
        cost_tolerance=1e-4,
        iterations=50)

    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
        opts_uncon=opts_ilqr,
        iterations=20,
        cost_tolerance=1.0e-5,
        cost_tolerance_intermediate=1.0e-4,
        constraint_tolerance=1e-3,
        penalty_scaling=10.,
        penalty_initial=100.)

    opts_altro = ALTROSolverOptions{T}(verbose=verbose,
        opts_al=opts_al,
        R_inf=1.0e-6,
        resolve_feasible_problem=false,
        projected_newton=false,
        projected_newton_tolerance=1.0e-3)

    solver = ALTROSolver(prob, opts_altro, infeasible=true)
end
#
# prob = gen_quad_flip()
# U0 = deepcopy(controls(prob))
# solver = iLQRSolver(prob, opts_ilqr)
# rollout!(solver)
# visualize!(vis, solver)
# solve!(solver)
# visualize!(vis, model, get_trajectory(solver))
#

# Try ALTRO
rpy = RPY{Float64}
cay = UnitQuaternion{Float64,CayleyMap}
quat = UnitQuaternion{Float64,IdentityMap}
solver = gen_quad_flip(cay)
Z0 = deepcopy(get_trajectory(solver))
# b3 = benchmark_solve!(solver)
solve!(solver)
max_violation(solver)
visualize!(vis, model, get_trajectory(solver))

function run_all(gen_prob;kwargs...)
    # Initial data structure
    data = Dict{Symbol,Vector}(:name=>Symbol[], :costfun=>Symbol[], :iterations=>Int[], :rotation=>Symbol[],
        :time=>Float64[], :cost=>Float64[])

    # Baseline methods
    rpy = RPY{Float64}
    cay = UnitQuaternion{Float64,CayleyMap}
    quat = UnitQuaternion{Float64,IdentityMap}
    println("Baseline methods")
    log_solve!(gen_prob(quat), data; kwargs...)
    # log_solve!(gen_prob(RodriguesParam{Float64}, use_rot=false), data; kwargs...)
    # log_solve!(gen_prob(MRP{Float64}, use_rot=false), data; kwargs...)
    log_solve!(gen_prob(rpy), data; kwargs...)

    # Quaternion Methods
    println("Quaternion methods")
    log_solve!(gen_prob(cay), data; kwargs...)
    return data
end
data_flip = run_all(gen_quad_flip, samples=10, evals=1)
visualize!(vis, model, get_trajectory(solver))
states(solver)[61]

waypoints!(vis, model, get_trajectory(solver),
    inds=[1,15,20,25,28,32,36,39,41,43,45,47,51,54,56,58,60,62,65,70,75,80,101])
delete!(vis["robot"])

############################################################################################
#                              COMBINED PLOT
############################################################################################
function convert_names(s::Symbol)
    if s == :IdentityMap
        return "Quat"
    elseif s == :RPY
        return "RPY"
    else
        return "iMLQR"
    end
end

df_flip = DataFrame(data_flip)
df_flip.problem = [:QuadFlip for i = 1:3]

df_br = DataFrame(data_br)
df_br.problem = [:BarrelRoll for i = 1:3]

df_zig = DataFrame(data_zig)
df_zig.problem = [:ZigZag for i = 1:3]

df = [df_zig; df_flip; df_br]
df.time_per_iter = df.time ./ df.iterations

legend = string.(unique(df.problem))
label = string.(unique(df.name))
coords_zig = Coordinates(string.(unique(df.name)), df_zig.time)
coords_flip = Coordinates(string.(unique(df.name)), df_flip.time)
coords_br = Coordinates(string.(unique(df.name)), df_br.time)

p = @pgf Axis(
    {
        ybar,
        ylabel="solve time (ms)",
        # "enlarge limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.07),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=label,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=0}"
    },
    Plot(coords_zig),
    Plot(coords_flip),
    Plot(coords_br),
    Legend(legend)
)

legend = convert_names.(unique(df.name))
label = string.(unique(df.problem))
df_quat = df[df.name .== :IdentityMap, :]
df_rpy = df[df.name .== :RPY, :]
df_cay = df[df.name .== :CayleyMap, :]

coords_quat = Coordinates(label, df_quat.time)
coords_rpy = Coordinates(label, df_rpy.time)
coords_cay = Coordinates(label, df_cay.time)

p = @pgf Axis(
    {
        ybar,
        ylabel="solve time (ms)",
        # "enlarge limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.07),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=label,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=0}"
    },
    Plot(coords_quat),
    Plot(coords_rpy),
    Plot(coords_cay),
    Legend(legend)
)
pgfsave("figs/timing_chart.tikz", p, include_preamble=false)

# Time per iteration

coords_quat = Coordinates(label, df_quat.time_per_iter)
coords_rpy = Coordinates(label, df_rpy.time_per_iter)
coords_cay = Coordinates(label, df_cay.time_per_iter)

p = @pgf Axis(
    {
        ybar,
        ylabel="time per iteration (ms)",
        # "enlarge limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.07),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=label,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=1}"
    },
    Plot(coords_quat),
    Plot(coords_rpy),
    Plot(coords_cay),
    Legend(legend)
)
