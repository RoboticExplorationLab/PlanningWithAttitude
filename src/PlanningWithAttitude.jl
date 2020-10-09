module PlanningWithAttitude

using StaticArrays
using LinearAlgebra
using Rotations
using RobotZoo
using Altro
using TrajectoryOptimization
import RobotDynamics
const RD = RobotDynamics
const TO = TrajectoryOptimization

# include("rotatedmodel.jl")
include("vecmodel.jl")
include("airplane_problem.jl")
include("quat_cons.jl")
include("quat_costs.jl")
include("quat_norm.jl")

export
    VecModel,
    YakProblems,
    QuatGeoCon,
    QuatErr,
    QuatVecEq,
    ErrorQuadratic,
    LieLQR



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
