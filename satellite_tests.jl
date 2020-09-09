using TrajectoryOptimization
const TO = TrajectoryOptimization
using TrajOptPlots

using JLD
using DataFrames
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using Plots
using ForwardDiff
using Random
using MeshCat
using Statistics
import LinearAlgebra: normalize

# Dynamics
model = Dynamics.Satellite2()
size(model)
x,u = rand(model)
dynamics(model,x,u)
N = 101
tf = 5.0

if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

angwrap(x) = min(x,360-x)

function final_angle_error(solver)
    q = Dynamics.orientation(solver.model, state(solver.Z[end]))
    qf = Dynamics.orientation(solver.model, solver.xf)
    rad2deg(angle(qf\q))
end

function total_angle(solver)
    θ = 0.0
    X = states(solver)
    model = solver.model
    for k = 1:solver.N-1
        q1 = Dynamics.orientation(model, X[k])
        q2 = Dynamics.orientation(model, X[k+1])
        θ += angle(q2\q1)
    end
    return rad2deg(θ)
end


function gen_sat_prob(Rot=UnitQuaternion{Float64,CayleyMap};
        ang=deg2rad(90), cost=:QuatLQR, use_rot=true)
    # Model
    model = Dynamics.Satellite2{Rot}(use_rot=use_rot)
    rot_dim = Rot <: UnitQuaternion ? 4 : 3
    N = 101
    tf = 5.0

    # Objective
    Q_diag = @SVector [1e-2,1e-2,1e-2, 0e-2,0e-2,0e-2,0e-2]
    R_diag = @SVector fill(1e-3,3)
    Q_rot = @SVector fill(1e-1, rot_dim)
    θ = ang
    u = normalize(-1 .+ 2*@SVector rand(3))
    # u = @SVector [1,0,0.]
    iω = @SVector [1,2,3]
    iq = @SVector [4,5,6,7]
    ωf = @SVector zeros(3)
    qf = expm(θ*u)
    xf = Dynamics.build_state(model, ωf, qf)

    costs = map(1:N) do k
        k < N ? s = 1 : s = 100
        if cost == :QuatLQR
            QuatLQRCost(Diagonal(Q_diag)*s, Diagonal(R_diag), xf)
        elseif cost == :SatDiff
            SatDiffCost(model, Diagonal(Q_diag[iω]),
                Diagonal(Q_rot[iω])*s, Diagonal(R_diag), SVector(qf), ωf)
        elseif cost == :Quadratic
            LQRCost(Diagonal([Q_diag[iω]; Q_rot])*s, Diagonal(R_diag), xf)
        end
    end
    obj = Objective(costs)

    # Initial Condition
    q0 = I(UnitQuaternion)
    x0 = Dynamics.build_state(model, (@SVector zeros(3)), q0)
    u0 = @SVector zeros(3)
    # Random.seed!(1)
    U0 = [u0 + randn(3)*4e-1 for k = 1:N-1]

    # Solver
    prob = Problem(model, obj, xf, tf, x0=x0)
    solver = iLQRSolver(prob)
    solver.opts.verbose = true
    initial_controls!(solver, U0)
    return solver
end

solver = gen_sat_prob(UnitQuaternion{Float64,CayleyMap}, cost=:QuatLQR,
    use_rot=true, ang=deg2rad(90))
solve!(solver)
visualize!(vis, solver.model, solver.Z)
solver.stats.iterations
final_angle_error(solver)
total_angle(solver)


data

function test_representation(Rot, costfun)
    data = Dict{Symbol,Vector}(:ang_err=>Float64[], :ang_total_err=>Float64[], :iterations=>Int64[], :success=>Bool[])
    for ang in [90,180,270]
        ang_true = angwrap(ang)
        use_rot = Rot <: UnitQuaternion
        solver = gen_sat_prob(Rot, cost=costfun, use_rot=false, ang=deg2rad(ang))
        solve!(solver)
        push!(data[:ang_err], final_angle_error(solver))
        total_ang = total_angle(solver)
        push!(data[:ang_total_err], abs(total_ang - ang_true))
        push!(data[:iterations], solver.stats.iterations)
        push!(data[:success], data[:ang_total_err][end] < 20 && data[:ang_err][end] < 2.0)
    end
    return data
end

data = test_representation(UnitQuaternion{Float64,VectorPart}, :Quadratic)
DataFrame(data)

data = Dict{Symbol,Any}(:ang=>Float64[], :cost=>Symbol[], :rotation=>Symbol[],
    :time=>Float64[], :iterations=>Int[], :ang_err=>Float64[], :ang_total=>Float64[])
run_set!(data, UnitQuaternion{Float64, ExponentialMap}, deg2rad(-90), cost=:SatDiff)
data

# Run Benchmarks
function log_solve!(data, solver, ang, func=median)
   # Run info
   push!(data[:ang], rad2deg(ang))
   push!(data[:cost], cost_type(solver))
   push!(data[:rotation], rot_type(solver))

   # Run stats
   b = time_solve(solver)
   push!(data[:time], time(func(b))*1e-6)  # ms
   push!(data[:iterations], solver.stats.iterations)
   push!(data[:ang_err], final_angle_error(solver))  # degree
   push!(data[:ang_total], total_angle(solver))  # degree

   data
end

function run_set!(data, Rot, ang; num_runs=10, kwargs...)
    for i = 1:num_runs
        solver = gen_sat_prob(Rot; ang=ang, kwargs...)
        log_solve!(data, solver, ang)
    end
end

data = Dict{Symbol,Any}(:ang=>Float64[], :cost=>Symbol[], :rotation=>Symbol[],
    :time=>Float64[], :iterations=>Int[], :ang_err=>Float64[], :ang_total=>Float64[])

function run_sim(data, θ, costfun, sets=[:quats, :baseline])

    if :quats ∈ sets
        # ExponentialMap
        println("Exponential Map")
        run_set!(data, UnitQuaternion{Float64, ExponentialMap}, θ, cost=costfun)

        # CayleyMap
        println("Cayley Map")
        run_set!(data, UnitQuaternion{Float64, CayleyMap}, θ, cost=costfun)

        # MRPMap
        println("MRP Map")
        run_set!(data, UnitQuaternion{Float64, MRPMap}, θ, cost=costfun)

        # VectorPart
        println("Vector Map")
        run_set!(data, UnitQuaternion{Float64, VectorPart}, θ, cost=costfun)
    end

    if :baseline ∈ sets
        # Normal Quaternion
        println("Quaternion")
        run_set!(data, UnitQuaternion{Float64, IdentityMap}, θ, cost=:Quadratic, use_rot=false)

        # Rodrigues Parameter
        println("Rodrigues parameters")
        run_set!(data, RodriguesParam{Float64}, θ, cost=:Quadratic, use_rot=false)

        # MRP
        println("MRPs")
        run_set!(data, MRP{Float64}, θ, cost=:Quadratic, use_rot=false)

        # RPY
        println("RPY Euler angles")
        run_set!(data, RPY{Float64}, θ, cost=:Quadratic, use_rot=false)
    end

    data
end

data = run_sim(data, deg2rad(270), :SatDiff)
@save "sat.jld" data

run_sim(data, deg2rad(270), :QuatLQR, [:quats])
data

# Add some calculated values
df = DataFrame(data)
df.ang_total_true = angwrap.(df.ang)
df.ang_err_total = @. abs(df.ang_total_true - df.ang_total)
df.angle = @. string(Int(df.ang))

total_ang_tol = 15
err_tol = 2
df.success = @. (df.ang_err_total < total_ang_tol) & (df.ang_err < err_tol)

df.time_per_iter = @. df.time / df.iterations
df.rots = short_names.(df.rotation)

percentage(x) = count(x) / length(x)


# Generate plots
df90 = df[(df.ang .≈ 90) .& (df.cost .!= :QuatLQR), :]
df270 = df[(df.ang .≈ 180) .& (df.cost .!= :QuatLQR), :]
df180 = df[(df.ang .≈ 270) .& (df.cost .!= :QuatLQR), :]
df270[df270.rots .== :Cay,:].ang_err


df_angs = (df90, df180, df270);
rotations = string.(unique(t.rots))

function bar_comparison(labels, data; err=zero(data), name="")
    coords = Coordinates(labels, data, yerror=err)
    @pgf Axis(
        {
            ybar,
            enlargelimits = 0.15,
            x="1.1cm",
            legend_style =
            {
                at = Coordinate(0.5, -0.15),
                anchor = "north",
                legend_columns = -1
            },
            symbolic_x_coords=labels,
            xtick = "data",
            nodes_near_coords,
            nodes_near_coords_align={vertical},
            "error bars/y dir=both",
            "error bars/y explicit",
        },
        Plot(coords),
        Legend([name])
    )
end

############################################################################################
# PLOT 1: Time per iteration
#       y-bar chart with an entry for each rotation type
#       average time per iteration across angles (90,270)
#       exlude 180 degrees since RPs die at 180 and never attempt to solve (cost blowup)
############################################################################################

df_no180 = df[.!isapprox.(df.ang, 180),:]
tip = by(df_no180, :rots, :time_per_iter => median)
tip_d = by(df_no180, :rots, :time_per_iter => std)
p = bar_comparison(rotations, tip.time_per_iter_median, err=tip_d.time_per_iter_std,
    name = "time per iteration (ms)")
pgfsave("figs/sat_time_per_iter.tikz", p, include_preamble=false)

############################################################################################
# PLOT 2: Iterations
#       y-bar chart with an entry for each rotation type
#       average number of iterattions for each of the 3 angles
############################################################################################
iter = map(df_angs) do df
    by(df, :rots, :iterations => median)
end
iter_d = map(df_angs) do df
    by(df, :rots, :iterations => std)
end

coords = map(1:3) do k
    Coordinates(rotations, iter[k].iterations_median, yerror=iter_d[k].iterations_std)
end

p = @pgf Axis(
    {
        ybar,
        ylabel=raw"\#iterations",
        # enlargelimits = 0.15,
        x="1.5cm",
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=rotations,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "error bars/y dir=both",
        "error bars/y explicit",
    },
    Plot(coords[1]),
    Plot(coords[2]),
    Plot(coords[3]),
    Legend(["90 degrees", "180 degrees", "270 degrees"])
)

pgfsave("figs/sat_iterations.tikz", p, include_preamble=false)


############################################################################################
# PLOT 3: Success rate
#       y-bar chart with an entry for each rotation type
#       success rate for each of the angles, and one cumulative
############################################################################################

succ = map(df_angs) do df
    by(df, :rots, :success => percentage)
end


succ = map(df_angs) do df
    by(df, :rots, :ang_err => median)
end

succ = map(df_angs) do df
    by(df, :rots, :ang_err_total => median)
end










coords = map(1:3) do k
    Coordinates(rotations, succ[k].success_percentage)
end

coords2 = map(1:8) do k
    Coordinates([90,180,270], [df.success_percentage[k] for df in succ])
end

p = @pgf Axis(
    {
        ybar,
        ylabel="success rate",
        # enlargelimits = 0.15,
        x="1.5cm",
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=rotations,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coords[1]),
    Plot(coords[2]),
    Plot(coords[3]),
    Legend(["90 degrees", "180 degrees", "270 degrees"])
)
pgfsave("figs/sat_success_rate.tikz", p, include_preamble=false)

p = @pgf Axis(
    {
        ybar,
        ylabel="success rate",
        "enlarge x limits" = 0.25,
        # x="1.5cm",
        x ="0.6mm",
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        # symbolic_x_coords=rotations,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coords2[1]),
    Plot(coords2[2]),
    Plot(coords2[3]),
    Plot(coords2[4]),
    Plot(coords2[5]),
    Plot(coords2[6]),
    Plot(coords2[7]),
    Plot(coords2[8]),
    Legend(rotations)
)
pgfsave("figs/sat_success_rate_byangle.tikz", p, include_preamble=false)
