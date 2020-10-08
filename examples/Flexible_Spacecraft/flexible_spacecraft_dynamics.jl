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

struct FlexSatellite <: LieGroupModel
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
Base.size(::FlexSatellite) = 17,4
Base.position(::FlexSatellite, x::SVector) = @SVector zeros(3)
orientation(::FlexSatellite, x::SVector) = UnitQuaternion(x[4], x[5], x[6], x[7])
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
