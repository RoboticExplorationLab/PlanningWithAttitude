module PlanningWithAttitude

using StaticArrays
using LinearAlgebra
using Rotations
import RobotDynamics
import TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization

include("rotatedmodel.jl")
include("vecmodel.jl")

export
    QuatSlackModel,
    UnitQuatConstraint,
    VecModel,
    QuatRenorm

# include("logger.jl")
# include("knotpoint.jl")
# include("models.jl")
# include("quaternions.jl")
# include("objective.jl")
# include("ilqr_solver.jl")
# include("ilqr.jl")
# include("attitude_costs.jl")

# include("rbstate.jl")
# include("plotting.jl")
# include("tracking_control.jl")
# include("lqr_monte_carlo.jl")
end
