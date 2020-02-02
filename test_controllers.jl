using TrajectoryOptimization
import TrajectoryOptimization.Dynamics.trim_controls
const TO = TrajectoryOptimization
using Parameters
using TrajOptPlots
using StaticArrays
using LinearAlgebra
using MeshCat
using BenchmarkTools
using JLD2
using DataFrames
using PlanningWithAttitude
import LinearAlgebra: normalize



############################################################################################
#                             SE3 QUADROTOR CONTROLLER
############################################################################################
Rot = UnitQuaternion{Float64,CayleyMap}
model = Dynamics.LeeQuad()
kx = 59.02
kv = 24.3
kR = 8.8
kO = 1.54
x0 = zeros(model)[1]
u0 = Dynamics.trim_controls(model)

tf = 10.0
dt = 1e-4
times = range(0,tf,step=dt)
Xref = [copy(x0) for k = 1:length(times)]
bref = [@SVector [1,0,0.] for k = 1:length(times)]

cntrl = SE3Tracking(model, Xref, bref, collect(times), kx=kx, kv=kv, kR=kR, kO=kO)
r = @SVector [1,0,0.]
qinit = expm(r * deg2rad(10))
qinit = TO.rotmat_to_quat(R0)
xinit = Dynamics.build_state(model, zeros(3), qinit, zeros(3), zeros(3))
X = simulate(model, cntrl, xinit, tf)
visualize!(vis, model, X[1:10:length(X)], tf)

R0 = @SMatrix [1 0 0;
      0 -0.9995 -0.0314;
      0 0.0314 -0.9995]
R0'
inv(R0)
model.J
model.mass

############################################################################################
#                                 LQR vs MLQR COMPARISON
############################################################################################
function build_controller(model,dt,Q_,R)
    Q = Diagonal(Dynamics.fill_error_state(model, Q_...))
    n,m = size(model)
    n = TO.state_diff_size(model)
    x0 = zeros(model)[1]
    u0 = trim_controls(model)
    cntrl = MLQR(model, dt, Q, R, x0, u0)
end

function build_SE3_controller(model,dt,tf)
    xref = zeros(model)[1]

    # Build reference trajectory
    times = range(0,tf,step=dt)
    Xref = [copy(xref) for k = 1:length(times)]
    bref = [@SVector [1,0,0.] for k = 1:length(times)]

    cntrl = SE3Tracking(model, Xref, bref, collect(times))
end

function build_LQR_controller(model,dt,Q_,R)
    Q = Diagonal(Dynamics.fill_state(model, Q_...))
    n,m = size(model)
    n = TO.state_diff_size(model)
    x0 = zeros(model)[1]
    u0 = trim_controls(model)
    cntrl = LQR(model, dt, Q, R, x0, u0)
end

dt = 0.01
tf = 10.0

xmin = @SVector [-1,-1,-1, 1,0,0,0, -0,-0,-0, -0,-0,-0]
xmax = @SVector [ 1, 1, 1, 1,0,0,0,  0, 0, 0,  0, 0, 0]
xmin = RBState(xmin)
xmax = RBState(xmax)
ICs = generate_ICs(model,xmin,xmax,10)

Rot = RodriguesParam{Float64}
Rot = MRP{Float64}
Rot = UnitQuaternion{Float64,ExponentialMap}
Rot = UnitQuaternion{Float64,CayleyMap}
Rot = UnitQuaternion{Float64,MRPMap}
model = Dynamics.LeeQuad(Rot)
xref = zeros(model)[1]
uref = trim_controls(model)

Q = Diagonal(Dynamics.fill_error_state(model,200,20,50,50.))
# Q = Diagonal(Dynamics.build_state(model, fill(200,3), (@SVector [50,50,500]), fill(50,3), [50,50,50.]))
R = Diagonal(@SVector fill(10.,4))
cntrl = MLQR(model, dt, Q, R, xref, uref)
# cntrl = build_controller(model,dt,Q_,R)
# cntrl = build_LQR_controller(model,dt,Q_,R)
# cntrl = build_SE3_controller(model, dt, tf)

r = @SVector [1,0,0.]
# xinit = Dynamics.build_state(model, zeros(3), expm(r * deg2rad(175)), zeros(3), zeros(3))
xinit = Dynamics.build_state(model,randbetween(xmin,xmax))
X = simulate(model, cntrl, xinit, tf, w=0)
visualize!(vis, model, X[1:10:length(X)], tf)
U = map(X) do x
    get_control(cntrl,x,0.)
end
err = map(X) do x
    state_diff(model, x, xref)
end
plot(U,legend=:bottom)
plot(err,4:6)
#=
Summary: Test with
LQR with RPY completely fails above 90°
LQR with Quats works
LQR with RP/MRP is identical to MLQR when the reference is 0 rotation
MLQR with RP works the best (same as CayleyMap)
Again RP rotates the correct way when the rotation > 180 (but breaks at 180)

Maybe try a bunch of different rotations and plot (heatmap) which ones fail?
=#
