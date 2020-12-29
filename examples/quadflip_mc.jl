import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using RobotDynamics
import RobotZoo.Quadrotor
using TrajectoryOptimization
using Altro
using BenchmarkTools
const TO = TrajectoryOptimization
const RD = RobotDynamics

using Random
using StaticArrays
using LinearAlgebra
using Rotations

using PlanningWithAttitude
using JLD2
using DataFrames

args = (
    integration = RK4,
    termcon = :quatvec,
    projected_newton = true,
    show_summary=true,
    verbose = 0,
    iterations = 100,
    static_bp = false 
)

## Solve problem
prob,opts = QuadFlipProblem(vecmodel=true, renorm=true; args...)
solver_vec = ALTROSolver(prob, opts, R_inf=1e-4, infeasible=true, static_bp=false)
prob,opts = QuadFlipProblem(vecmodel=false, renorm=true, costfun=QuatLQRCost; args...)
solver_err = ALTROSolver(prob, opts, R_inf=1e-4, infeasible=true, static_bp=false)
n,m = size(prob)
solve!(solver_vec)
solve!(solver_err)
visualize!(vis, Quadrotor(), get_trajectory(solver_vec))

## Get reference trajectories
Xvec = states(solver_vec)
Uvec = [u[SOneTo(4)] for u in controls(solver_vec)]
Xerr = states(solver_err)
Uerr = [u[SOneTo(4)] for u in controls(solver_err)]

## Solve with perturbed initial guess
prob,opts = QuadFlipProblem(vecmodel=true, renorm=true; args...)
solver_vec = ALTROSolver(prob, opts, R_inf=1e3, infeasible=true, projected_newton=false, iterations=400)
prob,opts = QuadFlipProblem(vecmodel=false, renorm=true, costfun=QuatLQRCost; args...)
solver_err = ALTROSolver(prob, opts, R_inf=1e3, infeasible=true, projected_newton=false, iterations=400)

## Run Monte-Carlo Analysis
T = 100 
iters = zeros(T,2)
status = fill(Altro.UNSOLVED,T,2) 
dt = [z.dt for z in prob.Z]
using Random
Random.seed!(1)

for i = 1:T
    println("Iteration $i")
    x_noise = [RBState(
        randn(3)*1.0, 
        expm(normalize(randn(3))*deg2rad(145.0)), 
        randn(3)*1.0, 
        randn(3)*1.0
    ) for k = 1:prob.N]
    u_noise = [randn(m)*0.10 for k = 1:prob.N-1]
    Zvec = Traj(RBState.(Xvec) .+ x_noise, Uvec .+ u_noise, dt)
    Zerr = Traj(RBState.(Xerr) .+ x_noise, Uerr .+ u_noise, dt)
    initial_trajectory!(solver_vec, Altro.infeasible_trajectory(TO.get_model(solver_vec), Zvec))
    initial_trajectory!(solver_err, Altro.infeasible_trajectory(TO.get_model(solver_err), Zerr))
    
    # Solve problem with perturbed initial guess
    solve!(solver_vec)
    solve!(solver_err)

    # Cache the iterations and solver status
    iters[i,1] = iterations(solver_vec)
    iters[i,2] = iterations(solver_err)
    status[i,1] = Altro.status(solver_vec)
    status[i,2] = Altro.status(solver_err)
end
println(sum(status .== Altro.SOLVE_SUCCEEDED, dims=1) ./ T)
@save joinpath(@__DIR__, "..", "flip_montecarlo.jdl2") iters status