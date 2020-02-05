module PlanningWithAttitude

using StaticArrays
using LinearAlgebra
using Parameters
using ForwardDiff
using TrajectoryOptimization
const TO = TrajectoryOptimization
using Statistics
using ControlSystems
using JLD2
# using Logging
# using Formatting

# include("logger.jl")
# include("knotpoint.jl")
# include("models.jl")
# include("quaternions.jl")
# include("objective.jl")
# include("ilqr_solver.jl")
# include("ilqr.jl")
# include("attitude_costs.jl")

include("rbstate.jl")
include("plotting.jl")
include("tracking_control.jl")
include("lqr_monte_carlo.jl")
end
