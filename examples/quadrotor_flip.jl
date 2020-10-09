import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using RobotDynamics
import RobotZoo.Quadrotor
using TrajectoryOptimization
using PlanningWithAttitude
using Altro
using TrajOptPlots
using BenchmarkTools
const TO = TrajectoryOptimization

using Random
using StaticArrays
using LinearAlgebra
using MeshCat
using Rotations

##
function gen_quad_flip(Rot=UnitQuaternion; slack::Bool=false, vecmodel::Bool=false, 
        renorm::Bool=false, costfun=LQRCost)
    model = Quadrotor{Rot}()
    if renorm
        model = QuatRenorm(model)
    end
    n,m = size(model)
    m += slack
    rsize = n-9

    # discretization
    N = 101 # number of knot points
    tf = 5.0
    dt = tf/(N-1) # total time

    # Initial condition
    x0_pos = @SVector [0., -1., 1.]
    x0 = RobotDynamics.build_state(model, x0_pos, UnitQuaternion(I), zeros(3), zeros(3))

    # cost
    Q_diag = RobotDynamics.build_state(model, 
        [1e-2,1e-2,5e-2], 
        fill(1e-5,rsize), 
        fill(1e-3,3), 
        fill(1e-2,3)
    )
    R = Diagonal(@SVector fill(1e-3,m))
    q_nom = UnitQuaternion(I)
    v_nom, ω_nom = zeros(3), zeros(3)
    x_nom = RobotDynamics.build_state(model, zeros(3), q_nom, v_nom, ω_nom)

    # waypoints
    ex = @SVector [1.0, 0, 0]
    wpts = [
        ((@SVector [0, 0.5, 1.5,]),   expm(ex*deg2rad(90))),
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

    """
    Costs
    """
    # intermediate costs
    Qw_diag = RobotDynamics.build_state(model, 
        [1e3,1e1,1e3], 
        (@SVector fill(5e4,rsize)), 
        fill(1,3), fill(10,3) 
    )
    Qf_diag = RobotDynamics.fill_state(model, 10., 100, 10, 10)
    xf = RobotDynamics.build_state(model, wpts[end][1], wpts[end][2], zeros(3), zeros(3))
    cost_nom = costfun(Diagonal(Q_diag), R, xf)

    # waypoint costs
    costs = map(1:length(wpts)) do i
        r,q = wpts[i]
        xg = RobotDynamics.build_state(model, r, q, v_nom, [2pi/3, 0, 0])
        if times[i] == N
            Q = Diagonal(Qf_diag)
            w = 10.
        else
            Q = Diagonal(1e-3*Qw_diag)
            w = 100.
        end
        costfun(Q, R, xg)
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
    conSet = ConstraintList(n,m,N)
    add_constraint!(conSet, GoalConstraint(xf, SA[1,2,3,8,9,10]), N:N)
    xmin = fill(-Inf,n)
    xmin[3] = 0.0
    bnd = BoundConstraint(n,m, x_min=xmin)
    add_constraint!(conSet, bnd, 1:N-1)

    if slack
        quad = QuatSlackModel(model)
        quatcon = UnitQuatConstraint(quad)
        add_constraint!(conSet, quatcon, 1:N-1)
    else
        quad = model
    end
    if vecmodel
        quad = VecModel(quad)
    end

    # Initialization
    u0 = zeros(quad)[2] 
    U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

    # Problem
    prob = Problem(quad, obj, xf, tf, x0=x0, constraints=conSet, integration=RK4)
    initial_controls!(prob, U_hover)

    # Infeasible start trajectory
    RobotDynamics.build_state(model, zeros(3), UnitQuaternion(I), zeros(3), zeros(3))
    X_guess = map(1:prob.N) do k
        t = (k-1)/prob.N
        x = (1-t)*x0 + t*xf
        RobotDynamics.build_state(model, position(model, x),
            expm(2pi*t*@SVector [1.,0,0]),
            RobotDynamics.linear_velocity(model, x),
            RobotDynamics.angular_velocity(model, x))
    end
    initial_states!(prob, X_guess)
    return prob
end

## Try ALTRO
opts = SolverOptions(
    cost_tolerance=1e-5,
    cost_tolerance_intermediate=1e-5,
    constraint_tolerance=1e-4,
    projected_newton_tolerance=1e-2,
    iterations_outer=40,
    penalty_scaling=10.,
    penalty_initial=0.1,
    show_summary=false,
    verbose=0
)
prob = gen_quad_flip(UnitQuaternion, slack=false, vecmodel=false, renorm=true,
    costfun=QuatLQRCost
)
solver = ALTROSolver(prob, opts, R_inf=1e-4, infeasible=true, show_summary=true)
solve!(solver)
visualize!(vis, quad, get_trajectory(solver))

##
prob = gen_quad_flip(UnitQuaternion, slack=false, vecmodel=false)
solver = ALTROSolver(prob, opts, R_inf=1e-3, infeasible=true, show_summary=false)
b1 = benchmark_solve!(solver)
iterations(solver)
Altro.print_summary(solver)

prob2 = gen_quad_flip(UnitQuaternion, slack=false, vecmodel=true)
solver2 = ALTROSolver(prob2, opts, R_inf=1e-3, infeasible=true)
b2 = benchmark_solve!(solver2)
Altro.print_summary(solver2)
iterations(solver2)
maximum([abs(1-norm(x[4:7])) for x in states(solver2)])
judge(minimum(b1), minimum(b2))

## Visualize
using TrajOptPlots, Blink
if !isdefined(Main, :vis)
    vis = Visualizer()
    open(vis, Blink.Window())
end
delete!(vis)
quad = Quadrotor()
TrajOptPlots.set_mesh!(vis, quad)
visualize!(vis, quad, get_trajectory(solver))
visualize!(vis, quad, get_trajectory(solver2))

## Plot y-z trajectory
using Plots
RobotDynamics.traj2(states(solver), xind=2, yind=3)

## Waypoints
TrajOptPlots.waypoints!(vis, quad, get_trajectory(solver), 
    inds=[1,15,20,25,28,30,32,35,40,42,44,46,48,50,52,54,56,58,60,62,65,75])

## Try Ipopt
using Ipopt, MathOptInterface
const MOI = MathOptInterface
prob = gen_quad_flip(UnitQuaternion, slack=false, vecmodel=true)
TO.add_dynamics_constraints!(prob)
nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)
optimizer = Ipopt.Optimizer(max_iter=1000, 
    tol=1e-4, constr_viol_tol=1e-4, dual_inf_tol=1e-4, compl_inf_tol=1e-4)
TO.build_MOI!(nlp, optimizer)
MOI.optimize!(optimizer)
visualize!(vis, quad, nlp.Z)

##
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

delete!(vis)
waypoints!(vis, model, get_trajectory(solver),
    inds=[1,15,20,25,28,32,36,39,41,43,45,47,51,54,56,58,60,62,65,70,75,80,101])
delete!(vis["robot"])

############################################################################################
#                              COMBINED PLOT
########################################    The linearization of the nonlinear discrete dynamics function $x_{k+1} = f(x_k,u_k)$ is 
    ``corrected'' using \ref{eq:quat_jacobian}:
    \begin{equation}
        A = E(f(x,u))^T \pdv{f}{x} E(x); \quad B = E(f(x,u))^T \pdv{f}{u}.
    \end{equation}
    Here we define the \textit{state attitude Jacobian} $E(x)$ to be a block-diagonal
    matrix where the block is an identity matrix for any vector-valued state and $G(q)$ for
    any quaternion in the state vector. 
    Using \eqref{eq:quat_gradient}, \eqref{eq:quat_hessian}, \eqref{eq:quat_jacobian} we 
    can derive similar modifications to the expansions of the cost and constraint functions.
    We refer the reader to the original ALTRO paper 
    \cite{howell2019altro} for futher details on the algorithm.
####################################################
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
