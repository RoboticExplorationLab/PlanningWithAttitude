
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
using Plots

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
ICs = generate_ICs(model,xmin,xmax,2000)
@time data2 = run_MC(ICs, dt_cntrl=0.01, dt=1e-3,
    types=[RPY{Float64}, CayleyMap, IdentityMap, SE3Tracking])

df = DataFrame(data2)
df.success = df.term_err .< 0.1
by(df, :name, :success=>percentage)
dfs = df[df.success, :]
by(dfs, :name, (:max_err=>mean, :max_err=>maximum, :avg_err=>mean))
df[.!df.success,:]

# x0 = zero(RBState)
# X = test_controller(ExponentialMap, ICs[1], dt_cntrl=0.01, tf=10.0)
# visualize!(vis, X[1:100:length(X)], tf)


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


#---------- Run Monte Carlo Analysis of Local Controllers ----------#
xmin = @SVector [-1,-1,-1, 1,0,0,0, -5,-5,-5, -5,-5,-5]
xmax = @SVector [ 1, 1, 1, 1,0,0,0,  5, 5, 5,  5, 5, 5]
Nsamples = 2000
us = map(1:Nsamples) do u
    ang = rand()*2pi - pi
    mag = pi*sqrt(rand())
    u = @SVector [mag*cos(ang), mag*sin(ang)]
    return u
end
y = [u[1] for u in us]
z = [u[2] for u in us]
scatter(y,z)
ICs = map(us) do u
    r = @SVector [1,0,0.]
    v = @SVector [0,0,0.]
    ω = @SVector [0,0,0.]
    q = expm(@SVector [u[1], 0, u[2]])
    RBState(r,q,v,ω)
end

@time data2 = run_MC(ICs, dt_cntrl=0.01, dt=1e-3,
    types=[CayleyMap, IdentityMap, RPY{Float64}, SE3Tracking])
df = DataFrame(data2)
df.success = df.term_err .< 0.2
df[df.name .== :RPY,:]
by(df, :name, :success=>percentage)

name = reverse(unique(df.name))

using LazySets
using PGFPlotsX
colors = Dict(:CayleyMap=>"blue", :SE3=>"green", :IdentityMap=>"red", :RPY=>"orange")
plots = map(name) do n
    inds = df[(df.name .== n) .& df.success, :IC]
    ninds = df[(df.name .== n) .& .!df.success, :IC]
    hull = convex_hull(us[inds])
    x = [rad2deg(h[1]) for h in hull]
    y = [rad2deg(h[2]) for h in hull]
    push!(x,x[1])
    push!(y,y[1])
    coord = Coordinates(x,y)
    @pgf Plot({"ultra thick",mark="none", color=colors[n]}, coord)
end;

points = Coordinates(rad2deg.(y),rad2deg.(z))

p = @pgf Axis({
    ylabel="yaw angle (deg)",
    xlabel="roll angle (deg)",
    xtick=-180:60:180,
    ytick=-180:60:180,
    legend_style = {
            at = Coordinate(0.5, -0.20),
            anchor = "north",
            legend_columns = -1
        },
    },
    plots...,
    PlotInc({"only marks",color="black",mark="*","mark_size"=0.6,"mark color=black"}, points),
    Legend(string.(name))
)
pgfsave("figs/mlqr_basin.tikz", p, include_preamble=false)
