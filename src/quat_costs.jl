using TrajectoryOptimization: QuadraticCostFunction, CostFunction
import RobotDynamics: state_dim, control_dim
import TrajectoryOptimization: is_blockdiag, is_diag, 
    stage_cost, gradient!, hessian!, change_dimension
import Base: +

function LieLQR(s::RD.LieState{Rot,P}, Q::Diagonal{<:Any,<:StaticVector}, R::Diagonal, 
        xf, uf=zeros(size(R,1)); kwargs...) where {Rot,P}
    if Rot <: UnitQuaternion && length(Q.diag) == 13
        Qd = deleteat(Q.diag, 4)
        Q = Diagonal(Qd)
    end
    n = length(s)
    n̄ = RD.state_diff_size(s) 
    G = zeros(n,n̄)
    RD.state_diff_jacobian!(G, s, xf)
    Q_ = G*Q*G' 
    # q = -Q_*xf
    # r = -R*uf
    # c = 0.5*xf'Q_*xf + 0.5*ur'R*uf
    LQRCost(Q_, R, xf, uf; kwargs...)
end

############################################################################################
#                        QUADRATIC QUATERNION COST FUNCTION
############################################################################################

struct DiagonalQuatCost{N,M,T,N4} <: QuadraticCostFunction{N,M,T}
    Q::Diagonal{T,SVector{N,T}}
    R::Diagonal{T,SVector{M,T}}
    q::SVector{N,T}
    r::SVector{M,T}
    c::T
    w::T
    q_ref::SVector{4,T}
    q_ind::SVector{4,Int}
    Iq::SMatrix{N,4,T,N4}
    function DiagonalQuatCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}},
            q::SVector{N,T}, r::SVector{M,T}, c::T, w::T,
            q_ref::SVector{4,T}, q_ind::SVector{4,Int}) where {T,N,M}
        Iq = @MMatrix zeros(N,4)
        for i = 1:4
            Iq[q_ind[i],i] = 1
        end
        Iq = SMatrix{N,4}(Iq)
        return new{N,M,T,N*4}(Q, R, q, r, c, w, q_ref, q_ind, Iq)
    end
end

state_dim(::DiagonalQuatCost{N,M,T}) where {T,N,M} = N
control_dim(::DiagonalQuatCost{N,M,T}) where {T,N,M} = M
is_blockdiag(::DiagonalQuatCost) = true
is_diag(::DiagonalQuatCost) = true

function DiagonalQuatCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}};
        q=(@SVector zeros(N)), r=(@SVector zeros(M)), c=zero(T), w=one(T),
        q_ref=(@SVector [1.0,0,0,0]), q_ind=(@SVector [4,5,6,7])) where {T,N,M}
    DiagonalQuatCost(Q, R, q, r, c, q_ref, q_ind)
end

function stage_cost(cost::DiagonalQuatCost, x::SVector, u::SVector)
    stage_cost(cost, x) + 0.5*u'cost.R*u + cost.r'u
end

function stage_cost(cost::DiagonalQuatCost, x::SVector)
    J = 0.5*x'cost.Q*x + cost.q'x + cost.c
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    J += cost.w*min(1+dq, 1-dq)
end

function gradient!(E::QuadraticCostFunction, cost::DiagonalQuatCost{T,N,M}, 
        x::SVector) where {T,N,M}
    Qx = cost.Q*x + cost.q
    q = x[cost.q_ind]
    dq = cost.q_ref'q
    if dq < 0
        Qx += cost.w*cost.Iq*cost.q_ref
    else
        Qx -= cost.w*cost.Iq*cost.q_ref
    end
    E.q .= Qx
    return false
end

function QuatLQRCost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}}, xf,
        uf=(@SVector zeros(M)); w=one(T), quat_ind=(@SVector [4,5,6,7])) where {T,N,M}
    r = -R*uf
    q = -Q*xf
    c = 0.5*xf'Q*xf + 0.5*uf'R*uf
    q_ref = xf[quat_ind]
    return DiagonalQuatCost(Q, R, q, r, c, w, q_ref, quat_ind)
end

function change_dimension(cost::DiagonalQuatCost, n, m, ix, iu)
    Qd = zeros(n)
    Rd = zeros(m)
    q = zeros(n)
    r = zeros(m)
    Qd[ix] = diag(cost.Q)
    Rd[iu] = diag(cost.R)
    q[ix] = cost.q
    r[iu] = cost.r
    qind = (1:n)[ix[cost.q_ind]]
    DiagonalQuatCost(Diagonal(Q_diag), Diagonal(R_diag), q, r, cost.c, cost.w,
        cost.q_ref, q_ind)
end

function (+)(cost1::DiagonalQuatCost, cost2::QuadraticCost)
    @assert state_dim(cost1) == state_dim(cost2)
    @assert control_dim(cost1) == control_dim(cost2)
    @assert norm(cost2.H) ≈ 0
    DiagonalQuatCost(cost1.Q + cost2.Q, cost1.R + cost2.R,
        cost1.q + cost2.q, cost1.r + cost2.r, cost1.c + cost2.c,
        cost1.w, cost1.q_ref, cost1.q_ind)
end

(+)(cost1::QuadraticCost, cost2::DiagonalQuatCost) = cost2 + cost1

function Base.copy(c::DiagonalQuatCost)
    DiagonalQuatCost(c.Q, c.R, c.q, c.r, c.c, c.w, c.q_ref, c.q_ind)
end


############################################################################################
#                             Error Quadratic
############################################################################################

struct ErrorQuadratic{Rot,N,M} <: CostFunction
    model::RD.RigidBody{Rot}
    Q::Diagonal{Float64,SVector{12,Float64}}
    R::Diagonal{Float64,SVector{M,Float64}}
    r::SVector{M,Float64}
    c::Float64
    x_ref::SVector{N,Float64}
    q_ind::SVector{4,Int}
end
function Base.copy(c::ErrorQuadratic)
    ErrorQuadratic(c.model, c.Q, c.R, c.r, c.c, c.x_ref, c.q_ind)
end

state_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = N
control_dim(::ErrorQuadratic{Rot,N,M}) where {Rot,N,M} = M

function ErrorQuadratic(model::RD.RigidBody{Rot}, Q::Diagonal{T,<:SVector{N0}},
        R::Diagonal{T,<:SVector{M}},
        x_ref::SVector{N}, 
        u_ref=(@SVector zeros(T,M)); 
        r=(@SVector zeros(T,M)), 
        c=zero(T),
        q_ind=(@SVector [4,5,6,7])
    ) where {T,N,N0,M,Rot}
    if Rot <: UnitQuaternion && N0 == N 
        Qd = deleteat(Q.diag, 4)
        Q = Diagonal(Qd)
    end
    r += -R*u_ref
    c += 0.5*u_ref'R*u_ref
    return ErrorQuadratic{Rot,N,M}(model, Q, R, r, c, x_ref, q_ind)
end


function stage_cost(cost::ErrorQuadratic, x::SVector)
    dx = RD.state_diff(cost.model, x, cost.x_ref, Rotations.ExponentialMap())
    return 0.5*dx'cost.Q*dx + cost.c
end

function stage_cost(cost::ErrorQuadratic, x::SVector, u::SVector)
    stage_cost(cost, x) + 0.5*u'cost.R*u + cost.r'u
end


function gradient!(E::QuadraticCostFunction, cost::ErrorQuadratic, x)
    f(x) = stage_cost(cost, x)
    ForwardDiff.gradient!(E.q, f, x)
    return false

    model = cost.model
    Q = cost.Q
    q = RD.orientation(model, x)
    q_ref = RD.orientation(model, cost.x_ref)
    dq = Rotations.params(q_ref \ q)
    err = RD.state_diff(model, x, cost.x_ref)
    dx = @SVector [err[1],  err[2],  err[3],
                    dq[1],   dq[2],   dq[3],   dq[4],
                   err[7],  err[8],  err[9],
                   err[10], err[11], err[12]]
    # G = state_diff_jacobian(model, dx) # n × dn

    # Gradient
    dmap = inverse_map_jacobian(model, dx) # dn × n
    # Qx = G'dmap'Q*err
    Qx = dmap'Q*err
    E.q = Qx
    return false
end
function gradient!(E::QuadraticCostFunction, cost::ErrorQuadratic, x, u)
    gradient!(E, cost, x)
    Qu = cost.R*u
    E.r .= Qu
    return false
end

function hessian!(E::QuadraticCostFunction, cost::ErrorQuadratic, x)
    f(x) = stage_cost(cost, x)
    ForwardDiff.hessian!(E.Q, f, x)
    return false

    model = cost.model
    Q = cost.Q
    q = RD.orientation(model, x)
    q_ref = RD.orientation(model, cost.x_ref)
    dq = Rotations.params(q_ref\q)
    err = RD.state_diff(model, x, cost.x_ref)
    dx = @SVector [err[1],  err[2],  err[3],
                    dq[1],   dq[2],   dq[3],   dq[4],
                   err[7],  err[8],  err[9],
                   err[10], err[11], err[12]]
    # G = state_diff_jacobian(model, dx) # n × dn

    # Gradient
    dmap = inverse_map_jacobian(model, dx) # dn × n

    # Hessian
    ∇jac = inverse_map_∇jacobian(model, dx, Q*err)
    # Qxx = G'dmap'Q*dmap*G + G'∇jac*G + ∇²differential(model, x, dmap'Q*err)
    Qxx = dmap'Q*dmap + ∇jac #+ ∇²differential(model, x, dmap'Q*err)
    E.Q = Qxx
    E.H .*= 0 
    return false
end

function hessian!(E::QuadraticCostFunction, cost::ErrorQuadratic, x, u)
    hessian!(E, cost, x)
    E.R .= cost.R
    return false
end


function change_dimension(cost::ErrorQuadratic, n, m)
    n0,m0 = state_dim(cost), control_dim(cost)
    Q_diag = diag(cost.Q)
    R_diag = diag(cost.R)
    r = cost.r
    if n0 != n
        dn = n - n0  # assumes n > n0
        pad = @SVector zeros(dn) # assume the new states don't have quaternions
        Q_diag = [Q_diag; pad]
    end
    if m0 != m
        dm = m - m0  # assumes m > m0
        pad = @SVector zeros(dm)
        R_diag = [R_diag; pad]
        r = [r; pad]
    end
    ErrorQuadratic(cost.model, Diagonal(Q_diag), Diagonal(R_diag), r, cost.c,
        cost.x_ref, cost.q_ind)
end

function (+)(cost1::ErrorQuadratic, cost2::QuadraticCost)
    @assert control_dim(cost1) == control_dim(cost2)
    @assert norm(cost2.H) ≈ 0
    @assert norm(cost2.q) ≈ 0
    if state_dim(cost2) == 13
        rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12,13]
        Q2 = Diagonal(diag(cost2.Q)[rm_quat])
    else
        Q2 = cost2.Q
    end
    ErrorQuadratic(cost1.model, cost1.Q + Q2, cost1.R + cost2.R,
        cost1.r + cost2.r, cost1.c + cost2.c,
        cost1.x_ref, cost1.q_ind)
end

(+)(cost1::QuadraticCost, cost2::ErrorQuadratic) = cost2 + cost1

@generated function state_diff_jacobian(model::RD.RigidBody{<:UnitQuaternion},
    x0::SVector{N,T}, errmap::D=Rotations.CayleyMap()) where {N,T,D}
    if D <: IdentityMap
        :(I)
    else
        quote
            q0 = RD.orientation(model, x0)
            # G = TrajectoryOptimization.∇differential(q0)
            G = Rotations.∇differential(q0)
            I1 = @SMatrix [1 0 0 0 0 0 0 0 0 0 0 0;
                        0 1 0 0 0 0 0 0 0 0 0 0;
                        0 0 1 0 0 0 0 0 0 0 0 0;
                        0 0 0 G[1] G[5] G[ 9] 0 0 0 0 0 0;
                        0 0 0 G[2] G[6] G[10] 0 0 0 0 0 0;
                        0 0 0 G[3] G[7] G[11] 0 0 0 0 0 0;
                        0 0 0 G[4] G[8] G[12] 0 0 0 0 0 0;
                        0 0 0 0 0 0 1 0 0 0 0 0;
                        0 0 0 0 0 0 0 1 0 0 0 0;
                        0 0 0 0 0 0 0 0 1 0 0 0;
                        0 0 0 0 0 0 0 0 0 1 0 0;
                        0 0 0 0 0 0 0 0 0 0 1 0;
                        0 0 0 0 0 0 0 0 0 0 0 1.]
        end
    end
end
function inverse_map_jacobian(model::RD.RigidBody{<:UnitQuaternion},
    x::SVector, errmap=Rotations.CayleyMap())
    q = RD.orientation(model, x)
    # G = TrajectoryOptimization.inverse_map_jacobian(q)
    G = Rotations.jacobian(inv(errmap), q)
    return @SMatrix [
            1 0 0 0 0 0 0 0 0 0 0 0 0;
            0 1 0 0 0 0 0 0 0 0 0 0 0;
            0 0 1 0 0 0 0 0 0 0 0 0 0;
            0 0 0 G[1] G[4] G[7] G[10] 0 0 0 0 0 0;
            0 0 0 G[2] G[5] G[8] G[11] 0 0 0 0 0 0;
            0 0 0 G[3] G[6] G[9] G[12] 0 0 0 0 0 0;
            0 0 0 0 0 0 0 1 0 0 0 0 0;
            0 0 0 0 0 0 0 0 1 0 0 0 0;
            0 0 0 0 0 0 0 0 0 1 0 0 0;
            0 0 0 0 0 0 0 0 0 0 1 0 0;
            0 0 0 0 0 0 0 0 0 0 0 1 0;
            0 0 0 0 0 0 0 0 0 0 0 0 1;
    ]
end

function inverse_map_∇jacobian(model::RD.RigidBody{<:UnitQuaternion},
    x::SVector, b::SVector, errmap=Rotations.CayleyMap())
    q = RD.orientation(model, x)
    bq = @SVector [b[4], b[5], b[6]]
    # ∇G = TrajectoryOptimization.inverse_map_∇jacobian(q, bq)
    ∇G = Rotations.∇jacobian(inv(errmap), q, bq)
    return @SMatrix [
        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 ∇G[1] ∇G[5] ∇G[ 9] ∇G[13] 0 0 0 0 0 0;
        0 0 0 ∇G[2] ∇G[6] ∇G[10] ∇G[14] 0 0 0 0 0 0;
        0 0 0 ∇G[3] ∇G[7] ∇G[11] ∇G[15] 0 0 0 0 0 0;
        0 0 0 ∇G[4] ∇G[8] ∇G[12] ∇G[16] 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0;
    ]
end

# function cost_expansion(cost::ErrorQuadratic{Rot}, model::AbstractModel,
#         z::KnotPoint{T,N,M}, G) where {T,N,M,Rot<:UnitQuaternion}
#     x,u = state(z), control(z)
#     model = cost.model
#     Q = cost.Q
#     q = orientation(model, x)
#     q_ref = orientation(model, cost.x_ref)
#     dq = SVector(q_ref\q)
#     err = state_diff(model, x, cost.x_ref)
#     dx = @SVector [err[1],  err[2],  err[3],
#                     dq[1],   dq[2],   dq[3],   dq[4],
#                    err[7],  err[8],  err[9],
#                    err[10], err[11], err[12]]
#     G = state_diff_jacobian(model, dx) # n × dn

#     # Gradient
#     dmap = inverse_map_jacobian(model, dx) # dn × n
#     Qx = G'dmap'Q*err
#     Qu = cost.R*u

#     # Hessian
#     ∇jac = inverse_map_∇jacobian(model, dx, Q*err)
#     Qxx = G'dmap'Q*dmap*G + G'∇jac*G + ∇²differential(model, x, dmap'Q*err)
#     Quu = cost.R
#     Qux = @SMatrix zeros(M,N-1)
#     return Qxx, Quu, Qux, Qx, Qu
# end

# function cost_expansion(cost::ErrorQuadratic, model::AbstractModel,
#         z::KnotPoint{T,N,M}, G) where {T,N,M}
#     x,u = state(z), control(z)
#     model = cost.model
#     q = orientation(model, x)
#     q_ref = orientation(model, cost.x_ref)
#     err = state_diff(model, x, cost.x_ref)
#     dx = err
#     G = state_diff_jacobian(model, dx) # n × n

#     # Gradient
#     dmap = inverse_map_jacobian(model, dx) # n × n
#     Qx = G'dmap'cost.Q*err
#     Qu = cost.R*u + cost.r

#     # Hessian
#     Qxx = G'dmap'cost.Q*dmap*G
#     Quu = cost.R
#     Qux = @SMatrix zeros(M,N)
#     return Qxx, Quu, Qux, Qx, Qu
# end
