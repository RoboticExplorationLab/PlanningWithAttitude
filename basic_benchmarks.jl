using TrajectoryOptimization
using StaticArrays
using BenchmarkTools, Statistics
using LinearAlgebra
using RExLabUtils
import RExLabUtils: boxplot

import TrajectoryOptimization: ∇rotate, ∇composition1, ∇composition2

# Generate Random Rotations
N = 1000  # number of rotations
qs = [rand(UnitQuaternion{Float64}) for i = 1:N]
ps = [MRP(q) for q in qs]
es = [RPY(q) for q in qs]

# Generate random vector
r = @SVector rand(3)

# Make sure multiplication and rotation is working the same for each representation
@assert qs[2]*qs[1]*r ≈ ps[2]*ps[1]*r ≈ es[2]*es[1]*r

# Statistical method for comparison
method = median


# Benchmark Single Composition
b1_q = @benchmark $qs[2]*$qs[1]
b1_p = @benchmark $ps[2]*$ps[1]
b1_e = @benchmark $es[2]*$es[1]
judge(method(b1_q), method(b1_p))
judge(method(b1_q), method(b1_e))
method(b1_e).time/method(b1_q).time
method(b1_p).time/method(b1_q).time

# Benchmark Many Compositions
b2_q = @benchmark foldr(*, $qs)
b2_p = @benchmark foldr(*, $ps)
b2_e = @benchmark foldr(*, $es)
judge(method(b2_q), method(b2_p))
judge(method(b2_q), method(b2_e))
method(b2_e).time/method(b2_q).time
method(b2_p).time/method(b2_q).time

# Benchmark Single Rotation
b3_q = @benchmark $qs[1]*$r
b3_p = @benchmark $ps[1]*$r
b3_e = @benchmark $es[1]*$r
judge(method(b3_q), method(b3_p))
judge(method(b3_q), method(b3_e))
method(b3_e).time/method(b3_q).time
method(b3_p).time/method(b3_q).time

# Benchmark Many Rotations
b4_q = @benchmark foldr(*, $qs, init=r)
b4_p = @benchmark foldr(*, $ps, init=r)
b4_e = @benchmark foldr(*, $es, init=r)
judge(method(b4_q), method(b4_p))
judge(method(b4_q), method(b4_e))
method(b4_e).time/method(b4_q).time
method(b4_p).time/method(b4_q).time


# Benchmark kinematics
ω = @SVector rand(3)
b5_q = @benchmark kinematics($qs[1], $ω)
b5_p = @benchmark kinematics($ps[1], $ω)
b6_q = @benchmark map(q->kinematics(q,$ω), qs)
b6_p = @benchmark map(q->kinematics(q,$ω), ps)
j5 = judge(method(b5_q), method(b5_p))
j6 = judge(method(b6_q), method(b6_p))
1/j5.ratio.time
1/j6.ratio.time


# Benchmark rotmat
@assert map(rotmat, qs) ≈ map(rotmat, ps) ≈ map(rotmat, es)
b7_q = @benchmark map(rotmat, $qs)
b7_p = @benchmark map(rotmat, $ps)
b7_e = @benchmark map(rotmat, $es)
j7_p = judge(method(b4_q), method(b4_p))
j7_e = judge(method(b4_q), method(b4_e))
1/j7_p.ratio.time
1/j7_e.ratio.time

# Benchmark ∇rotate
@btime ∇rotate($qs[1], $r)
@btime ∇rotate($ps[1], $r)
@btime ∇rotate($es[1], $r)
function ∇rotate(res::Vector, qs::Vector, r)
    for i in eachindex(qs)
        res[i] = ∇rotate(qs[i], r)
    end
end
res_q = map(q->∇rotate(q,r), qs)
res_p = map(q->∇rotate(q,r), ps)

b8_q = @benchmark ∇rotate($res_q, $qs, $r)
b8_p = @benchmark ∇rotate($res_p, $ps, $r)
b8_e = @benchmark ∇rotate($res_p, $es, $r)
judge(method(b8_q), method(b8_p))
judge(method(b8_q), method(b8_e))
method(b8_e).time/method(b8_q).time
method(b8_p).time/method(b8_q).time

# Benchmark ∇composition1
@btime ∇composition1($qs[2], $qs[1])
@btime ∇composition1($ps[2], $ps[1])
@btime ∇composition1($es[2], $es[1])
function ∇composition1(res::Vector, qs::Vector)
    q0 = qs[1]
    for i in eachindex(qs)
        res[i] = ∇composition1(qs[i], q0)
    end
end
res_q = map(q->∇composition1(q,qs[1]), qs)
res_p = map(q->∇composition1(q,ps[1]), ps)

b9_q = @benchmark ∇composition1($res_q, $qs)
b9_p = @benchmark ∇composition1($res_p, $ps)
b9_e = @benchmark ∇composition1($res_p, $es)
judge(method(b9_q), method(b9_p))
judge(method(b9_q), method(b9_e))
method(b9_e).time/method(b9_q).time
method(b9_p).time/method(b9_q).time

# Benchmark ∇composition2
@btime ∇composition2($qs[2], $qs[1])
@btime ∇composition2($ps[2], $ps[1])
@btime ∇composition2($es[2], $es[1])
function ∇composition2(res::Vector, qs::Vector)
    q0 = qs[1]
    for i in eachindex(qs)
        res[i] = ∇composition2(qs[i], q0)
    end
end
res_q = map(q->∇composition2(q,qs[1]), qs)
res_p = map(q->∇composition2(q,ps[1]), ps)

b10_q = @benchmark ∇composition2($res_q, $qs)
b10_p = @benchmark ∇composition2($res_p, $ps)
b10_e = @benchmark ∇composition2($res_p, $es)
judge(method(b10_q), method(b10_p))
judge(method(b10_q), method(b10_e))
method(b10_e).time/method(b10_q).time
method(b10_p).time/method(b10_q).time

b10_q.times

function whiskers(data)
    lo,up = quantile(data,(0.25,0.75))
    iq = up-lo
    return lo-1.5*iq, up+1.5*iq
end

function boxplot2(x...; labels="")

    if isempty(labels)
        labels = [string(i) for i = 1:length(x)]
    end
    label = join(labels, ", ")

    no_coords = Coordinates(zeros(0), zeros(0))
    plots = @pgf map(x) do data
        wlo, wup = whiskers(data)
        lo,up = quantile(data, (0.25,0.75))
        PlotInc({
        "boxplot prepared" = {
                "lower whisker" = wlo,
                "lower quartile" = lo,
                "median" = median(data),
                "upper quartile" = up,
                "upper whisker" = wup
            },
        "fill", "draw=black", "mark=none"}, no_coords)
    end

    ymax = -Inf
    ymin = Inf
    for data in x
        lo,up = whiskers(data)
        ymin = min(ymin,lo)
        ymax = max(ymax,up)
    end
    @show ymin*0.9, ymax*1.1

    p = @pgf Axis(
        {
            "boxplot/draw direction=y",
            "restrict y to domain=0:1E6",
            "x axis line style={opacity=0}",
            "axis x line*=bottom",
            "axis y line=left",
            "enlarge y limits",
            "xtick" = 1:length(x),
            "xticklabels" = label,
        },
        plots...
    )
end
t = b10_q.times
whiskers(t)
data = randn(300)
boxplot2(data)
std(data) + mean(data)

whiskers(data)

quantile(data, range(0,stop=1,length=5))
nquantile(data, 4)
quantile(data,1)
maximum(data)
boxplot(rand(30), rand(3), rand(100), labels=["A","B","C"])
boxplot(rand(30), rand(3), rand(100))
boxplot2(b10_q.times, b10_p.times)


using PGFPlotsX


print_tex(p)
print_tex(Table(["row setp=\\, yindex=0"], data))
t = @pgf Table({ "x index" = 2, "y index" = 1 }, randn(10, 3))
t1 = @pgf Table({ "y index=0"}, randn(3,1))
t2 = @pgf Table({ "y index=0"}, randn(3,1))
print_tex(t)
Coordinates(zeros(0),zeros(0))

t1_q = median(b8_q.times)/1e3
t1_p = median(b8_p.times)/1e3

t2_q = median(b9_q.times)/1e3
t2_p = median(b9_p.times)/1e3

t3_q = median(b10_q.times)/1e3
t3_p = median(b10_p.times)/1e3

tests = ["rotate", "comp1", "comp2"]
q_times = [t1_q, t2_q, t3_q]
q_err = [std(b8_q.times), std(b9_q.times), std(b10_q.times)] / 1e3
p_times = [t1_p, t2_p, t3_p]
p_err = [std(b8_p.times), std(b9_p.times), std(b10_p.times)] /1e3

p = @pgf Axis(
    {
        ybar,
        "height = 10cm",
        "width = 10cm",
        enlargelimits = 0.15,
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        ylabel = raw"\#participants",
        symbolic_x_coords=tests,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    # Plot(Coordinates([("rotate", t1_q), ("comp1", t2_q), ("comp2", t3_q)])),
    PlotInc({"error bars/y dir=both", "error bars/y explicit"}, q_coords),
    PlotInc({"error bars/y dir=both", "error bars/y explicit"}, p_coords),
    # Plot(Coordinates([("tool8", 4), ("tool9", 4), ("tool10", 4)])),
    # Plot(Coordinates([("tool8", 1), ("tool9", 1), ("tool10", 1)])),
    Legend(["Quat", "MRP"])
)

print_tex(p)

c = Coordinates([("rotate", t1_q), ("comp1", t2_q), ("comp2", t3_q)])
Coordinates(1:3, rand(3), yerrorplus=rand(3)/4, yerrorminus=rand(3)/4)
q_coords = map(zip(tests, q_times, q_err)) do (test, t, e)
    Coordinate((test, t), error=(0,e))
end
q_coords = Coordinates(q_coords)

p_coords = map(zip(tests, p_times, p_err)) do (test, t, e)
    Coordinate((test, t), error=(0,e))
end
p_coords = Coordinates(p_coords)

function grouped_bars(bars; errors=[], legend_names="", group_names="")
    n_groups = length(bars)
    n_series = length(bars[1])
    @assert all(length.(bars) .== n_series)
    if !isempty(errors)
        @assert all(length.(errors) .== n_series)
        @assert length(errors) == n_groups
    end

    # Create Coordinates
    plots = map(1:n_series) do i

    end
end
bars = [(t1_q, t1_p), (t2_q, t2_p), (t3_q, t3_p)]
errs = collect(zip(q_err, p_err))
grouped_bars(bars, errors=errs, legend_names=("Quat", "MRP"), group_names=tests)

Coordinates()
