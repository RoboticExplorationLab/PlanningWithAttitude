
using TrajectoryOptimization
using ForwardDiff
using Random
using StaticArrays
using LinearAlgebra
using MeshCat
using TrajOptPlots
using Rotations
using Plots
using Quaternions
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

model = Dynamics.Quadrotor2{UnitQuaternion{Float64,CayleyMap}}()
# model = Dynamics.Quadrotor2{UnitQuaternion{Float64,MRPMap}}()
# model = Dynamics.Quadrotor2{UnitQuaternion{Float64,IdentityMap}}()
n,m = 13,4

if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

# discretization
N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1) # total time

# Initial condition
x0 = Dynamics.build_state(model, [0,-1,1], I(UnitQuaternion), zeros(3), zeros(3))

# Final condition
xf = Dynamics.build_state(model, [0,1,1], -I(UnitQuaternion), zeros(3), zeros(3))


# Intermediate frames
qs = [UnitQuaternion(Quat(RotZ(ang))) for ang in range(0,2pi,length=N)]
qs = [ExponentialMap(ang*@SVector [0,0,1.]) for ang in range(0,pi,length=N)]
ys = range(x0[2],xf[2],length=N)
X_guess = map(1:N) do k
    pos = [x0[1], ys[k], x0[3]]
    q = qs[k]
    Dynamics.build_state(model, pos, q, zeros(3), [0,0,2pi/tf])
end


# Build Objective
Q_diag = Dynamics.build_state(model,
    fill(1e-3,3), (@SVector fill(0e-1,4)), fill(1e-3,3), fill(1e-1,3))
R_diag = @SVector fill(1e-4, 4)
no_quat = SVector{12}(deleteat!(collect(1:n), 4))
G = TO.state_diff_jacobian(model, X_guess[21])
zero_quat = @SVector [1,1,1, 0,0,0,0, 1,1,1, 1,1,1]
plot(X_guess,4:7)


costs = map(1:N) do k
    x = X_guess[k]
    Q = Diagonal(Q_diag)
    R = Diagonal(R_diag)
    LQRCost(Q,R,x, checks=false)
    QuatLQRCost(Q,R,x)
end
obj = Objective(costs)


# Solve Problem
Random.seed!(2)
u0 = @SVector fill(0.5*9.81/4, m)
udiff = -2e-2*@SVector [-1,-0, 1,0.]
U0 = [copy(u0) + udiff + randn(4)*0e-2 for k = 1:N-1]
prob = Problem(model, obj, xf, tf, x0=x0)
solver = iLQRSolver(prob, opts_ilqr)
solver.opts.verbose = true
initial_controls!(solver, U0)
rollout!(solver)
visualize!(vis, model, get_trajectory(solver))


solver.opts.bp_reg_initial = 1.0
solver.opts.bp_reg_min = 0.1
initial_controls!(solver, U0)
solve!(solver)

controls(solver)[10]
u0

# set_mesh!(vis, model)
visualize!(vis, model, get_trajectory(solver))

k = 50
ix = solver.Z[k]._x
iu = solver.Z[k]._u
solver.Q.uu[k]
solver.S.xx[k+1]
fdx = solver.G[k+1]'solver.∇F[k][ix,ix]*solver.G[k]
fdu = solver.G[k+1]'solver.∇F[k][ix,iu]
Quu =      solver.Q.uu[k]      + fdu'solver.S.xx[k+1]*fdu
fdu'solver.S.xx[k+1]*fdu
fdu'fdu
solver.S.xx[k+2][4:6,4:6]
fdx

fdu

# costfun = QuatLQRCost(Diagonal(Q_diag), Diagonal(R_diag), xf)
# x,u = rand(model)
# xn = Dynamics.flipquat(model, x)
# Qx,Qu = TO.gradient(costfun, x, u)
# Qx_n,Qu_n = TO.gradient(costfun, xn, u)
# Qu ≈ Qu_n
# ForwardDiff.gradient(x->TO.stage_cost(costfun, x, u), x) ≈ Qx
# ForwardDiff.gradient(u->TO.stage_cost(costfun, x, u), u) ≈ Qu
# ForwardDiff.gradient(x->TO.stage_cost(costfun, x, u), xn) ≈ Qx
#
# Qxx,Quu,Qux = TO.hessian(costfun, x, u)
# ForwardDiff.hessian(x->TO.stage_cost(costfun, x, u), x) ≈ Qxx
# ForwardDiff.hessian(u->TO.stage_cost(costfun, x, u), u) ≈ Quu
#

x,u = rand(model)
xn = Dynamics.flipquat(model, x)
costfun = RBCost(model, Diagonal(Q_diag[no_quat]), Diagonal(R_diag), x)
TO.stage_cost(costfun, x, u)
TO.stage_cost(costfun, xn, u)
TO.hessian(costfun, x, u)

Qx,Qu = TO.gradient(costfun, x, u)
Qx_n,Qu_n = TO.gradient(costfun, xn, u)
Qx ≈ Qx_n
Qu ≈ Qu_n
Qx1 = ForwardDiff.gradient(x->TO.stage_cost(costfun, x, u), x)
ForwardDiff.gradient(u->TO.stage_cost(costfun, x, u), u) ≈ Qu
Qx2 = ForwardDiff.gradient(x->TO.stage_cost(costfun, x, u), xn)
Qx1
Qx2
