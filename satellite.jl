using PlanningWithAttitude
using StaticArrays
using LinearAlgebra
using BenchmarkTools

# Dynamics
model = Satellite()
N = 101
tf = 5.0
dt = 0.1

x,u = rand(model)
z = KnotPoint(x,u,dt)
discrete_dynamics(model, x, u, 0.1)
dynamics(model, x, u)
discrete_jacobian(model, z)

# Objective
Q_diag = @SVector [1e-2,1e-2,1e-2, 1e-2,1e-2,1e-2,1e-2]
R_diag = @SVector fill(1e-4,3)
xf = @SVector [0,0,0, sqrt(2)/2, sqrt(2)/2, 0, 0]
costfun = LQRCost(Q_diag, R_diag, xf)
obj = Objective(costfun, N)

x0 = @SVector [0,0,0, 1,0,0,0.]
U0 = [@SVector zeros(3) for k = 1:N-1]

# Solver
solver = iLQRSolver(model, obj, x0, tf, N)
initial_controls!(solver, U0)
rollout!(solver)
cost(obj, solver.Z)

initial_controls!(solver, U0)
solve!(solver)


vis = Visualizer(); open(vis);
set_mesh!(vis, model)
visualize!(vis, model, solver.Z)
