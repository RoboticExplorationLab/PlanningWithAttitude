
using TrajectoryOptimization
import TrajectoryOptimization.Dynamics.trim_controls
const TO = TrajectoryOptimization
using PlanningWithAttitude
using Parameters
using TrajOptPlots
using StaticArrays
using LinearAlgebra
using MeshCat
using BenchmarkTools
using JLD2
using DataFrames
import LinearAlgebra: normalize
import TrajectoryOptimization: state_diff
import PlanningWithAttitude: generate_ICs, run_MC

model = Dynamics.Quadrotor2{MRP{Float64}}()
if !isdefined(Main,:vis)
    vis = Visualizer()
    open(vis)
    set_mesh!(vis, model)
end

percentage(x) = count(x) / length(x)

#---------- Run Monte Carlo Analysis of Local Controllers ----------#
xmin = @SVector [-1,-1,-1, 1,0,0,0, -5,-5,-5, -5,-5,-5]
xmax = @SVector [ 1, 1, 1, 1,0,0,0,  5, 5, 5,  5, 5, 5]
xmin = RBState(xmin)
xmax = RBState(xmax)
ICs = generate_ICs(model,xmin,xmax,100)
data2 = run_MC(ICs, dt_cntrl=0.01)

df = DataFrame(data2)
df.success = df.term_err .< 0.1
by(df, :name, :success=>percentage)
dfs = df[df.success, :]
by(dfs, :name, (:max_err=>mean, :max_err=>maximum, :avg_err=>mean))

x0 = zero(RBState)
X = test_controller(ExponentialMap, ICs[1], dt_cntrl=0.01, tf=10.0)
visualize!(vis, X[1:100:length(X)], tf)


# Calculate controller speed
function get_controls(X, cntrl)
    for x in X
        PlanningWithAttitude.get_control(cntrl, x, 0.0)
    end
end
dt = 0.01
tf = 10

model = Dynamics.Quadrotor2()
Q = Diagonal(Dynamics.fill_error_state(model, 200,200,50,50.))
R = Diagonal(@SVector fill(1.0, 4))
xref,uref = zeros(model)[1], Dynamics.trim_controls(model)
cntrl = MLQR(model, 0.01, Q, R, xref, uref)
xinit = Dynamics.build_state(model, ICs[1])
X = simulate(model, cntrl, xinit, 10.0, w=0.0)
b1 = @benchmark get_controls($X, $cntrl)

times = range(0,tf,step=dt)
Xref = [copy(xref) for k = 1:length(times)]
bref = [@SVector [1,0,0.] for k = 1:length(times)]
Xdref = [@SVector zeros(13) for k = 1:length(times)]
cntrl = SE3Tracking(model, Xref, Xdref, bref, collect(times))

b2 = @benchmark get_controls($X, $cntrl)
using Statistics
j1 = judge(median(b1), median(b2))
1/ratio(j1).time
