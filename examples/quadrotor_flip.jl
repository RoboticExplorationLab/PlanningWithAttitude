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

## Try ALTRO
# prob = gen_quad_flip(UnitQuaternion, slack=false, vecmodel=false, renorm=true,
#     costfun=QuatLQRCost
# )
prob,opts = QuadFlipProblem(UnitQuaternion, slack=false, vecmodel=false, renorm=true,
    costfun=QuatLQRCost
)
solver = ALTROSolver(prob, opts, R_inf=1e-4, infeasible=true, show_summary=true)
solve!(solver)

## Original Method
prob2 = gen_quad_flip(UnitQuaternion, slack=false, vecmodel=true)
prob2,opts = QuadFlipProblem(UnitQuaternion, slack=false, vecmodel=true)
solver2 = ALTROSolver(prob2, opts, R_inf=1e-4, infeasible=true, show_summary=true)
solve!(solver2)

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
delete!(vis)
TrajOptPlots.waypoints!(vis, quad, get_trajectory(solver), 
    inds=[1,15,20,25,28,30,32,35,40,42,44,46,48,50,52,54,56,58,60,62,65,68,70,75,90,100],
    color=HSL(colorant"green"), color_end=HSL(colorant"red"))
