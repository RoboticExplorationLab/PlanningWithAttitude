const RD = RobotDynamics
# post processing
function post_process(solver)
    X = states(solver)
    U = controls(solver)
    X = Vector.(X)
    U = Vector.(U)

    pts = length(X)-1
    q_hist = zeros(4,pts)
    ω_hist = zeros(3,pts)
    u_hist = zeros(4,pts)
    η_hist = zeros(pts)
    ηd_hist = zeros(pts)
    θ_hist = zeros(pts)
    r_hist = zeros(4,pts)
    for i = 1:pts
        ω_hist[:,i] = X[i][1:3]
        q_hist[:,i] = X[i][4:7]
        η_hist[i] = X[i][8]
        ηd_hist[i] = X[i][9]
        r_hist[:,i] = X[i][10:13]
        u_hist[:,i] = U[i][1:4]

        θ_hist[i] = 2*atan(norm(q_hist[2:4,i]),q_hist[1,i])
    end

    return ω_hist, q_hist, u_hist,η_hist, ηd_hist, r_hist, θ_hist
end

import RobotDynamics: dynamics, forces, moments, wrenches, mass_matrix, inertia, inertia_inv, orientation
import RobotDynamics: state_dim, control_dim




import Rotations: lmult, vmat, hmat

struct FlexSatellite <: RD.LieGroupModel
    J::SArray{Tuple{3,3},Float64,2,9}
    B::SArray{Tuple{3,4},Float64,2,12}
    C::SArray{Tuple{3,3},Float64,2,9}
    K::SArray{Tuple{3,3},Float64,2,9}
    δ::SArray{Tuple{3,3},Float64,2,9}
    first_inv::SArray{Tuple{3,3},Float64,2,9}
    ϕMΣ::SArray{Tuple{3,3},Float64,2,9}
end


# units of kg-m
ϕMΣ  = [0 1 0;
       1 0 0;
       0 .2 -.8];

ϕMΣ = SMatrix{3,3}(ϕMΣ)


# units of kg-m^2
δ =  [0 0 1;
         0 1 0;
        -.7 .1 .1]
δ = copy(transpose(δ))
δ = SMatrix{3,3}(δ)

J = diagm([1;2;3])

J = SMatrix{3,3}(J)

mass = 28.54*14.5939

B = @SMatrix [0.965926  0  -0.965926  0 ;
        0.258819 -0.258819 0.258819 -0.258819;
        0  0.965926 0 -0.965926]

zeta = [.001;.001;.001]
Delta = [.05; .2; .125] * (2*pi)

# damping and stiffness matrices
C = zeros(3,3)
K = zeros(3,3)
for i =1:3
    C[i,i] = 2*zeta[i]*Delta[i];
    K[i,i] = Delta[i]^2;
end

C = SMatrix{3,3}(C)
K = SMatrix{3,3}(K)

first_inv = inv(J - δ*δ')
FlexSatellite() = FlexSatellite(J,B,C,K,δ,first_inv,ϕMΣ)
RobotDynamics.control_dim(::FlexSatellite) = 4
# Base.size(::FlexSatellite) = 17,4
Base.position(::FlexSatellite, x::SVector) = @SVector zeros(3)
RD.orientation(::FlexSatellite, x::SVector) = UnitQuaternion(x[4], x[5], x[6], x[7])
RobotDynamics.LieState(::FlexSatellite) = RobotDynamics.LieState(UnitQuaternion{Float64}, (3,10))
function dynamics(model::FlexSatellite, x::SVector, u::SVector,t)
    ω = @SVector [x[1], x[2], x[3]]
    q = normalize(@SVector [x[4], x[5], x[6], x[7]])
    η = @SVector [x[8],x[9],x[10]]
    η_dot = @SVector [x[11],x[12],x[13]]
    r = @SVector [x[14],x[15],x[16],x[17]]
    J = model.J
    B = model.B
    C = model.C
    K = model.K
    δ = model.δ
    first_inv = model.first_inv
    ϕMΣ = model.ϕMΣ



    τ = @SVector zeros(3)
    F = @SVector zeros(3)

    ωdot = first_inv*(τ -B*u/60 -
        cross(ω,J*ω + δ*η_dot + B*r) +
        δ*(C*η_dot + K*η + ϕMΣ*F))
    qdot = 0.5*lmult(q)*hmat()*ω
    η_ddot = -δ'*ωdot -C*η_dot - K*η - ϕMΣ*F
    rdot = u/60
    return [ωdot; qdot;η_dot;η_ddot;rdot]
end

function RD.discrete_dynamics(::Type{Q}, model::FlexSatellite, x::SVector, u::SVector, t, dt) where Q <: RD.Explicit
    x2 = RD.integrate(Q, model, x, u, t, dt)
    return x2
    ω = x2[SA[1,2,3]]
    q = normalize(x[SA[4,5,6,7]]) 
    x_ = x[SA[8,9,10,11,12,13,14,15,16,17]]
    return [ω; q; x_]
end


## Problem definition
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
	SA[ dot(dcm_from_q(x[4:7])*con.bodyvec,con.keepoutdir)-cosd(con.keepoutangle)]
)


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

function SatelliteKeepOutProblem(constrained::Bool=true; vecstate::Bool=false, costfun=LQRCost)
	# Discretization
	tf = 100.0
	N = 401
	dt = tf/(N-1)

	# Model
	model = FlexSatellite()
	n,m = size(model)

	# Initial and final states
	ω = @SVector zeros(3)
	q0 = Rotations.params(expm(deg2rad(150) * normalize(@SVector [1,2,3])))
	qf = Rotations.params(UnitQuaternion(I))
	r0 = zeros(4)
	x0 = [ω; q0;zeros(3);zeros(3);r0]
	xf = SVector{17}([ω; qf;zeros(3);zeros(3);r0])

	# Objective
	Q = Diagonal(@SVector [10,10,10, 1,1,1,1, 10,10.0,10, 10,10,10,1,1,1,1.])
	R = Diagonal(@SVector fill(10.0, m))
	Qf = Q * N
	cost0 = costfun(Q, R, xf, w=0.1)
	cost_term = costfun(Qf, R, xf, w=1.0)
	obj = Objective(cost0, cost_term, N)
	# obj = LQRObjective(Q,R,Qf,xf,N)

	# constaint
	cons =  ConstraintList(n,m,N)
	bnd = BoundConstraint(n,m, u_min=-.5, u_max=.5)
	add_constraint!(cons,bnd,1:N-1)
	# add_constraint!(cons, QuatVecEq(n, UnitQuaternion(qf), SA[4,5,6,7]), N)
	add_constraint!(cons, QuatVecEq(n, UnitQuaternion(qf), SA[4,5,6,7]), N)

	# keep out
	keepoutdir = @SVector [1.0,0,0]
	bodyvec = @SVector [0.360019,-0.92957400,0.07920895]
	keepout = AttitudeKeepOut( n, keepoutdir, bodyvec, 40.0 )

	if constrained
		add_constraint!(cons,keepout,1:N)
	end

	if vecstate
		model = VecModel(model)
	end

	## Solve
	prob = Problem(model, obj, xf, tf, x0=x0,constraints = cons,N=N;integration=RD.RK4)
	# U0 = [@SVector randn(m) for k = 1:N-1] .* 1e-1
	# initial_controls!(prob, U0)

	opts = SolverOptions(
		verbose=1,
		constraint_tolerance = 1e-5,
		penalty_scaling = 200.0,
		penalty_initial = 1e3,
		cost_tolerance = 1e-4,
		cost_tolerance_intermediate = 1e-4,
		projected_newton_tolerance = 1e-4,
		show_summary = true,
		projected_newton = false,
		verbose_pn = 1
	)
	return prob, opts
end