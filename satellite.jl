using PlanningWithAttitude
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using Plots
using ForwardDiff
import PlanningWithAttitude: cost_gradient, stage_cost
include("visualization.jl")


# Dynamics
model = Satellite()
N = 101
tf = 5.0

if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

# Objective
Q_diag = @SVector [1e-3,1e-3,1e-3, 1e-2,1e-2,1e-2,1e-2]
R_diag = @SVector fill(1e-4,3)
θs = range(0,2pi,length=N)
u = normalize([1,0,0])
iω = @SVector [1,2,3]
costs = map(1:N) do k
    θ = θs[k]
    ωf = [2pi/5,0,0]
    qf = @SVector [cos(θ/2), u[1]*sin(θ/2), u[2]*sin(θ/2), u[3]*sin(θ/2)]
    qf = qf * rand([-1,1])
    xf = @SVector [ωf[1], ωf[2], ωf[3], qf[1], qf[2], qf[3], qf[4]]
    k < N ? s = 1 : s = 1
    # LQRCost(Q_diag*s, R_diag, xf)
    SatCost(Diagonal(Q_diag[iω]), Diagonal(R_diag), qf, 1.0, ωf)
end

Xref = map(1:N) do k
    θ = θs[k] #+ deg2rad(185)
    qf = @SVector [cos(θ/2), u[1]*sin(θ/2), u[2]*sin(θ/2), u[3]*sin(θ/2)]
    xf = @SVector [2pi/5,0,0, qf[1], qf[2], qf[3], qf[4]]
end
# plot(Xref,4:7)
# set_states!(Z, Xref)
# set_controls!(Z, U0 .* 0)
# cost(obj, Z)
# PlanningWithAttitude.stage_cost(costs[1], Z[1])
# PlanningWithAttitude.cost_gradient(solver, costs[101], Z[101])
# grad = [cost_gradient(solver, costs[k], Z[k])[1] for k = 1:N]
# plot(Array(hcat(grad...))')

x,u = rand(model)
z = KnotPoint(x,u,0.1)
G = PlanningWithAttitude.state_diff_jacobian(model, x)
grad = ForwardDiff.gradient(x->stage_cost(costs[1],x,u),x)
cost_gradient(solver, costs[1], z)[1] ≈ G'grad
gra


θ = θs[end]
xf = @SVector [0,0,0, cos(θ/2), u[1]*sin(θ/2), u[2]*sin(θ/2), u[3]*sin(θ/2)]
obj = Objective(costs, N)

# Initial Condition
x0 = @SVector [2pi/5,0,0, 1,0,0,0.]
u0 = @SVector [0.0, 0, 0]
U0 = [@SVector randn(3) for k = 1:N-1] .* 1e-1

# Solver
solver = iLQRSolver(model, obj, x0, tf, N)
initial_controls!(solver, U0)
rollout!(solver)
cost(obj, solver.Z)

initial_controls!(solver, U0)
# initial_controls!(solver, controls(Z_sol))
solver.opts.verbose = true
solve!(solver)
solver.stats.iterations
states(solver)[end][4:7]'xf[4:7]
visualize!(vis, model, solver.Z)
Z = solver.Z

plot(controls(solver))

cost(obj, Z)
obj.J
plot(obj.J, label="unwind", title="cost")
cost(obj, Z_sol)
plot!(obj.J, label="correct")

grad_rx1 = [PlanningWithAttitude.cost_gradient(solver, costs[k], Z[k])[1][4] for k = 1:N]
grad_rx2 = [PlanningWithAttitude.cost_gradient(solver, costs[k], Z_sol[k])[1][4] for k = 1:N]
plot(grad_rx1, label="unwind", title="Gradient")
plot!(grad_rx2, label="correct")

cost(obj, Z)
plot(grad_rx1 .* 50, label="gradient")
plot!(obj.J*10, label="cost")
q1 = [z.z[4] for z in Z]
plot!(q1, label="qx")
plot!(rad2deg.(2 .* acos.(q1)), label="angle")


plot!(controls(Z) .* 0.1, 1:1)
plot(controls(Z_sol), 1:1)

plot(states(Z),4:7)
plot(states(Z_sol),4:7)


Z_sol = deepcopy(solver.Z)

@btime begin
    initial_controls!($solver, $U0)
    solve!($solver)
end



visualize!(vis, model, solver.Z)
