export
    SatCost

struct SatCost <: CostFunction
    Q::Diagonal{Float64,SVector{3,Float64}}
    R::Diagonal{Float64,SVector{3,Float64}}
    q::SVector{3,Float64}
    q0::SVector{4,Float64}
    Qq::Float64
    c::Float64
end

function SatCost(Q::Diagonal,R::Diagonal,q0::SVector, Qq=0.0, ω0=@SVector zeros(3))
    q = -Q*ω0
    c = 0.5*ω0'Q*ω0
    SatCost(Q,R,q,q0,Qq,c)
end

stage_cost(cost::SatCost, z::KnotPoint) = stage_cost(cost, state(z), control(z))
function stage_cost(cost::SatCost, x::SVector, u::SVector)
    ω = @SVector [x[1], x[2], x[3]]
    q = @SVector [x[4], x[5], x[6], x[7]]
    J = 0.5*(ω'cost.Q*ω + u'cost.R*u) + cost.q'ω + cost.c
    d = cost.q0'q
    if d ≥ 0
        J += 1-d
    else
        J += 1+d
    end
    return J
end

function cost_gradient(solver::iLQRSolver, cost::SatCost, z::KnotPoint)
    Q = cost.Q
    u = control(z)
    ω = @SVector [z.z[1], z.z[2], z.z[3]]
    q = @SVector [z.z[4], z.z[5], z.z[6], z.z[7]]
    d = cost.q0'q
    Qω = Q*ω + cost.q
    if d ≥ 0
        Qq = -cost.q0'Lmult(q)*Vmat()'
    else
        Qq =  cost.q0'Lmult(q)*Vmat()'
    end
    Qx = [Qω; Qq']
    Qu = cost.R*u
    return Qx, Qu
end

function cost_hessian(solver::iLQRSolver, cost::SatCost, z::KnotPoint{T,N,M}) where {T,N,M}
    Q = cost.Q
    q = @SVector [z.z[4], z.z[5], z.z[6], z.z[7]]
    q0 = cost.q0
    Qxx = Diagonal(@SVector [Q[1,1], Q[2,2], Q[3,3], 0,0,0])
    mycost(q::SVector{4}) = min(1-q0'q, 1+q0'q)
    mycost(g::SVector{3}) = mycost(Lmult(q)*cayley_map(g))
    g = @SVector zeros(3)
    Qqq = ForwardDiff.hessian(mycost, g)
    d = q0'q
    if d ≤ 0
        d *= -1
    end
    Qqq = I(3)*d
    Qxx = @SMatrix [
        Q[1,1] 0 0 0 0 0;
        0 Q[2,2] 0 0 0 0;
        0 0 Q[3,3] 0 0 0;
        0 0 0 Qqq[1] Qqq[4] Qqq[7];
        0 0 0 Qqq[2] Qqq[5] Qqq[8];
        0 0 0 Qqq[3] Qqq[6] Qqq[9];
    ]
    Quu = cost.R
    Qux = @SMatrix zeros(M,N-1)
    return Qxx, Quu, Qux
end
