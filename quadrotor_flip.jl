
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

function gen_quad_flip(; costfun=:QuatLQR)
    model = Dynamics.Quadrotor2{UnitQuaternion{Float64,VectorPart}}()
    n,m = size(model)

    # discretization
    N = 101 # number of knot points
    tf = 5.0
    dt = tf/(N-1) # total time

    # Initial condition
    x0_pos = @SVector [0., -1., 1.]
    x0 = Dynamics.build_state(model, x0_pos, I(UnitQuaternion), zeros(3), zeros(3))

    # cost
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
            ((@SVector [0, 0.5, 1.0]),    expm(ex*deg2rad(360))),
            ((@SVector [0, 1.0, 1.0]),    expm(ex*deg2rad(360))),
            ]
    # times = [35, 41, 47, 51, 55, 61, 70, 101]
    times = [45, 51, 55, 75, 101]
    Qw_diag = Dynamics.build_state(model, [1e3,1e1,1e3], sq*(@SVector fill(1e3,4)), fill(1,3), fill(10,3))
    Qf_diag = Dynamics.fill_state(model, 10., 100*sq, 10, 10)
    xf = Dynamics.build_state(model, wpts[end][1], wpts[end][2], zeros(3), zeros(3))

    if costfun == :QuatLQR
        cost_nom = QuatLQRCost(Diagonal(Q_diag), R, xf, w=0.01)
    else
        cost_nom = LQRCost(Diagonal(Q_diag), R, xf)
    end

    costs = map(1:length(wpts)) do i
        r,q = wpts[i]
        xg = Dynamics.build_state(model, r, q, v_nom, [2pi/3, 0, 0])
        if times[i] == N
            Q = Diagonal(Qf_diag)
            w = 100.
        else
            Q = Diagonal(1e-3*Qw_diag)
            w = 1000.
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
    add_constraint!(conSet, GoalConstraint(xf), N:N)
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
    model = Dynamics.Quadrotor2{UnitQuaternion{Float64, CayleyMap}}()
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
    initial_controls!(prob, U_hover)
    prob
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

# Try ALTRO (not working)
x0 = prob.x0
xf = prob.xf
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
visualize!(vis, model, Z)
prob = gen_quad_flip()
initial_states!(prob, X_guess)

opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    cost_tolerance=1e-4,
    iterations=50)

opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=10,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=1e-3,
    penalty_scaling=10.,
    penalty_initial=100.)

opts_altro = ALTROSolverOptions{T}(verbose=verbose,
    opts_al=opts_al,
    R_inf=1.0e-4,
    resolve_feasible_problem=false,
    projected_newton=false,
    projected_newton_tolerance=1.0e-3)
solver = ALTROSolver(prob, opts_altro, infeasible=true)
solve!(solver)
max_violation(solver)

visualize!(vis, model, get_trajectory(solver))
states(solver)[61]
