import TrajectoryOptimization.Dynamics.trim_controls
using TrajectoryOptimization
const TO = TrajectoryOptimization
using TrajOptPlots
using StaticArrays
using LinearAlgebra
using MeshCat

model = Dynamics.Quadrotor2{MRP{Float64}}()
if !isdefined(Main,:vis)
    vis = Visualizer()
    open(vis)
    set_mesh!(vis, model)
end


Q0 = 1
R0 = 1

function build_controller(model)
    n,m = size(model)
    n = TO.state_diff_size(model)
    Q = Diagonal(@SVector fill(Q0, n))
    R = Diagonal(@SVector fill(R0, m))
    x0 = zeros(model)[1]
    u0 = trim_controls(model)
    cntrl = MLQR(model, x0, u0, 0.1, Q, R)
end


model = Dynamics.Quadrotor2{RPY{Float64}}()
n,m = size(model)
n = TO.state_diff_size(model)
Q = Diagonal(@SVector fill(Q0, n))
R = Diagonal(@SVector fill(R0, m))
x0 = zeros(model)[1]
u0 = trim_controls(model)
cntrl = LQR(model, x0, u0, 0.1, Q, R)

# LQR (Quat)
model = Dynamics.Quadrotor2{UnitQuaternion{Float64,IdentityMap}}()
n,m = size(model)
x0 = zeros(model)[1]
u = trim_controls(model)
Q = Diagonal(@SVector fill(Q0, n))
R = Diagonal(@SVector fill(R0, m))
cntrl = LQR(model, x0, u0, 0.1, Q, R)
Kquat = cntrl.K

# MLQR (MRP)
model = Dynamics.Quadrotor2{MRP{Float64}}()
x0 = @SVector zeros(12)
cntrl = MLQR(model, x0, u0, 0.1, Q, R)

# MLQR (RP)
model = Dynamics.Quadrotor2{RodriguesParam{Float64}}()
x0 = @SVector zeros(12)
cntrl = MLQR(model, x0, u0, 0.1, Q, R)


# MLQR (Quaternion)
model = Dynamics.Quadrotor2{UnitQuaternion{Float64,MRPMap}}()
n,m = size(model)
n = TO.state_diff_size(model)
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(1.0, m))
x0 = zeros(model)[1]
cntrl = MLQR(model, x0, u0, 0.1, Q, R)

Rot = UnitQuaternion{Float64,VectorPart}
Rot = RodriguesParam{Float64}
Rot = MRP{Float64}
model = Dynamics.Quadrotor2{Rot}()
cntrl = build_controller(model)

r = @SVector [1,0,0.]
xinit = Dynamics.build_state(model, zeros(3), expm(r * deg2rad(210)), zeros(3), zeros(3))
X = simulate(model, cntrl, xinit, 6.0)
visualize!(vis, model, X[1:10:length(X)], 6.0)

#=
Summary: Test with
LQR with RPY completely fails above 90°
LQR with Quats works
LQR with RP/MRP is identical to MLQR when the reference is 0 rotation
MLQR with RP works the best (same as CayleyMap)
Again RP rotates the correct way when the rotation > 180 (but breaks at 180)

Maybe try a bunch of different rotations and plot (heatmap) which ones fail?
=#
