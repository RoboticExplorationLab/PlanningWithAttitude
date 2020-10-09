cd(dirname(@__FILE__))
Pkg.activate(".")
using MATLAB
using StaticArrays
using Parameters
using RobotDynamics
using Rotations
using LinearAlgebra
# using Plots
using TrajectoryOptimization
using Altro


include(joinpath(dirname(@__FILE__),"flexible_spacecraft_dynamics.jl"))

const TO = TrajectoryOptimization


# Discretization
tf = 100.0
N = 401
dt = tf/(N-1)

# Model
model = FlexSatellite()
n,m = size(model)

# quaternion to dcm functions
function skew(v)
	"""Skew-symmetric matrix from 3 element array"""
    return @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
end
function dcm_from_q(q)
	"""DCM from quaternion, scalar first"""
	s = @views q[1]
	v = @views q[2:4]
    return I + 2*skew(v)*(s*I + skew(v))
end


struct AttitudeKeepOut{T} <: TO.StateConstraint
	n::Int
	keepoutdir::SVector{3,T}
	bodyvec::SVector{3,T}
	keepoutangle::Float64
	function AttitudeKeepOut(n::Int, keepoutdir::SVector, bodyvec::SVector, keepoutangle::T) where T
		new{T}(n,keepoutdir,bodyvec,keepoutangle)
	end
end

TO.state_dim(con::AttitudeKeepOut) = con.n
TO.sense(::AttitudeKeepOut) = Inequality()
Base.length(::AttitudeKeepOut) = 1
TO.evaluate(con::AttitudeKeepOut, x::SVector) = (
SA[ dot(dcm_from_q(x[4:7])*con.bodyvec,con.keepoutdir)-cosd(con.keepoutangle)])

# Initial and final states
ω = @SVector zeros(3)
q0 = Rotations.params(expm(deg2rad(150) * normalize(@SVector [1,2,3])))
qf = Rotations.params(UnitQuaternion(I))
r0 = zeros(4)
x0 = [ω; q0;zeros(3);zeros(3);r0]
xf = [ω; qf;zeros(3);zeros(3);r0]

# Objective
Q = Diagonal(@SVector [10,10,10,100,100,100,100,10,10.0,10,10,10,10,0,0,0,0])
R = Diagonal(@SVector fill(1.0, m))
Qf = Q
obj = LQRObjective(Q,R,Qf,xf,N)

# constaint
cons =  ConstraintList(n,m,N)
bnd = BoundConstraint(n,m, u_min=-.5, u_max=.5)
add_constraint!(cons,bnd,1:N-1)

## keep out
keepoutdir = @SVector [1.0,0,0]
bodyvec = @SVector [0.360019,-0.92957400,0.07920895]
keepout = AttitudeKeepOut( n, keepoutdir, bodyvec, 40.0 )

add_constraint!(cons,keepout,1:N)



prob = Problem(model, obj, xf, tf, x0=x0,constraints = cons,N=N;integration=RK4)

solver = ALTROSolver(prob)
solver.opts.projected_newton = true
solver.solver_al.opts.verbose = true
set_options!(solver, iterations = 4000)
set_options!(solver,constraint_tolerance = 1e-4)
solve!(solver)

println(iterations(solver))


ω_hist, q_hist, u_hist,η_hist, ηd_hist, r_hist,θ_hist = post_process(solver)

keepout_truth = zeros(size(q_hist,2))
g_hist = zeros(3,size(q_hist,2))
for i = 1:size(q_hist,2)
	# keepout_truth[i] = dot(dcm_from_q(q_hist[:,i])*bodyvec,keepoutdir) - cosd(20.0)
	keepout_truth[i] = acosd(dot(dcm_from_q(q_hist[:,i])*bodyvec,keepoutdir))
	q = q_hist[:,i]
	g_hist[:,i] = q[2:4]/q[1]
end

mat"
figure
hold on
title('Pointing Error')
ylabel('Pointing Error (Degrees)')
xlabel('Time (s)')
plot(0.0:$dt:(($N-2)*$dt),rad2deg($θ_hist))
hold off
"

mat"
figure
hold on
ylabel('Torque (Nm)')
xlabel('Time (s)')
title('Controls')
plot(0.0:$dt:(($N-2)*$dt),$u_hist')
hold off
"

mat"
figure
hold on
title('Angle from Sun (Constrained)')
a = area([0 100],[40 40]);
a(1).FaceColor = [1 0.8 0.8];
plot((0:($N-2))*$dt,$keepout_truth,'b','linewidth',2)
legend('Keep Out Zone','Trajectory')
xlabel('Time (s)')
ylabel('Angle from Sun (Degrees)')
hold off
"

mat"
f = @(x,y,z) -(3657152068667140*x^2 + 16745716480053206*x*y - 1426901590817633*x*z + 10142677805652334*y^2 - 1426901590817633*y + 10142677805652334*z^2 - 16745716480053206*z + 3657152068667140)/(9007199254740992*(x^2 + y^2 + z^2 + 1))

endind = 200
figure
hold on
title('Rodrigues Parameter Trajectory History')
plot3($g_hist(1,:),$g_hist(2,:),$g_hist(3,:),'b','linewidth',2)
%plot3(g_hist_uncon(1,1:endind),g_hist_uncon(2,1:endind),g_hist_uncon(3,1:endind),'r','linewidth',2)
%plot3(g_hist_con(1,1:endind),g_hist_con(2,1:endind),g_hist_con(3,1:endind),'b','linewidth',2)
fimplicit3(f,'FaceColor',[.5 .5 .5])
xlim([-.5 1.1])
ylim([-.2 2.1])
zlim([-.5 3])
legend('Constrained Trajectory','Constraint Surface')

xlabel('g_1')
ylabel('g_2')
zlabel('g_3')
% axis equal
hold off
"

# g_hist_con = copy(g_hist)
# file = matopen("g_hist_con.mat", "w")
# write(file, "g_hist_con", g_hist_con)
# close(file)
