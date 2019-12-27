using TrajectoryOptimization
using StaticArrays
using BenchmarkTools, Statistics
using LinearAlgebra

import TrajectoryOptimization: ∇rotate, ∇composition1, ∇composition2

# Generate Random Rotations
N = 1000  # number of rotations
qs = [rand(UnitQuaternion) for i = 1:N]
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

# Benchmark Single Composition
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
b8_q = @benchmark map(q->∇rotate(q,$r), qs)
b8_p = @benchmark map(q->∇rotate(q,$r), ps)
b8_e = @benchmark map(q->∇rotate(q,$r), es)
judge(method(b8_q), method(b8_p))
judge(method(b8_q), method(b8_e))
method(b8_e).time/method(b8_q).time
method(b8_p).time/method(b8_q).time

# Benchmark ∇composition1
@btime ∇composition1($qs[2], $qs[1])
@btime ∇composition1($ps[2], $ps[1])
@btime ∇composition1($es[2], $es[1])
b9_q = @benchmark map(q->∇composition1(q,$qs[1]), qs)
b9_p = @benchmark map(q->∇composition1(q,$ps[1]), ps)
b9_e = @benchmark map(q->∇composition1(q,$es[1]), es)
judge(method(b9_q), method(b9_p))
judge(method(b9_q), method(b9_e))
method(b9_e).time/method(b9_q).time
method(b9_p).time/method(b9_q).time

# Benchmark ∇composition2
@btime ∇composition2($qs[2], $qs[1])
@btime ∇composition2($ps[2], $ps[1])
@btime ∇composition2($es[2], $es[1])
b10_q = @benchmark map(q->∇composition2(q,$qs[1]), qs)
b10_p = @benchmark map(q->∇composition2(q,$ps[1]), ps)
b10_e = @benchmark map(q->∇composition2(q,$es[1]), es)
judge(method(b10_q), method(b10_p))
judge(method(b10_q), method(b10_e))
method(b10_e).time/method(b10_q).time
method(b10_p).time/method(b10_q).time
