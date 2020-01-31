import TrajectoryOptimization.Dynamics.trim_controls
using TrajectoryOptimization
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

n,m = size(model)
u0 = Dynamics.trim_controls(model)
x0 = @SVector zeros(12)

A,B = linearize(model, x0, u0, 0.1)
Q = Diagonal(@SVector fill(10.0, n))
R = Diagonal(@SVector fill(1.0, m))

cntrl = LQR(model, x0, u0, 0.1, Q, R)

get_control(cntrl, x0, 0)
ex = @SVector [1,0,0.]
xinit = Dynamics.build_state(model, [0.1,0,0], MRP(expm(ex*deg2rad(10))), zeros(3), [0,0,0])
X = simulate(model, cntrl, xinit, 6.0)
visualize!(vis, model, X[1:10:length(X)], 6.0)
