import Pkg; Pkg.activate(@__DIR__)
# using MATLAB
using StaticArrays
using Parameters
using RobotDynamics
using Rotations
using LinearAlgebra
# using Plots
using TrajectoryOptimization
using PlanningWithAttitude
using Altro


const TO = TrajectoryOptimization

using PlanningWithAttitude: post_process, dcm_from_q

##
prob,opts = SatelliteKeepOutProblem(vecstate=true, costfun=LQRCost)
prob,opts = SatelliteKeepOutProblem(vecstate=false, costfun=QuatLQRCost)
solver = ALTROSolver(prob, opts)
solve!(solver)


##
bodyvec = prob.constraints[3].bodyvec
keepoutdir = prob.constraints[3].keepoutdir
ω_hist, q_hist, u_hist,η_hist, ηd_hist, r_hist,θ_hist = post_process(solver)

keepout_truth = zeros(size(q_hist,2))
g_hist = zeros(3,size(q_hist,2))
for i = 1:size(q_hist,2)
	# keepout_truth[i] = dot(dcm_from_q(q_hist[:,i])*bodyvec,keepoutdir) - cosd(20.0)
	keepout_truth[i] = acosd(dot(dcm_from_q(q_hist[:,i])*bodyvec,keepoutdir))
	q = q_hist[:,i]
	g_hist[:,i] = q[2:4]/q[1]
end

using Plots
plot(keepout_truth)
plot(controls(solver))

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
