export
    Objective,
    QuadraticCost,
    LQRCost,
    cost

abstract type CostFunction end

struct Objective{C<:CostFunction}
    costs::Vector{C}
    J::Vector{Float64}
end

function Objective(costs::Vector{<:CostFunction}, N::Int)
    return Objective(costs, zeros(N))
end

function Objective(cost::CostFunction, N::Int)
    costs = [cost for k = 1:N]
    return Objective(costs, zeros(N))
end

function Objective(cost::CostFunction, costterm::CostFunction, N::Int)
    costs = [cost for k = 1:N-1]
    push!(costs, costterm)
    return Objective(costs, zeros(N))
end

@inline get_J(obj::Objective) = obj.J


function cost!(obj::Objective, Z::Traj)
    N = length(Z)
    J = obj.J
    for k in eachindex(obj.costs)
        k == N ? dt = 1.0 : dt = Z[k].dt
        J[k] = stage_cost(obj.costs[k], Z[k])*dt
    end
end

function cost(obj::Objective, Z::Traj)
    cost!(obj, Z)
    return sum(obj.J)
end


struct QuadraticCost{N,M} <: CostFunction
    Q::Diagonal{Float64,SVector{N,Float64}}
    R::Diagonal{Float64,SVector{M,Float64}}
    q::SVector{N,Float64}
    r::SVector{M,Float64}
    c::Float64
end

function stage_cost(cost::QuadraticCost, z::KnotPoint)
    x,u = state(z), control(z)
    return 0.5*(x'cost.Q*x + u'cost.R*u) + cost.q'x + cost.r'u + cost.c
end

function gradient(cost::QuadraticCost, z::KnotPoint)
    x,u = state(z), control(z)
    Qx = cost.Q*x + cost.q
    Qu = cost.R*u + cost.r
    return Qx, Qu
end

function hessian(cost::QuadraticCost, z::KnotPoint{T,N,M}) where {T,N,M}
    x,u = state(z), control(z)
    Qxx = cost.Q
    Quu = cost.R
    Qux = @SMatrix zeros(M,N)
    return Qxx, Quu, Qux
end

function LQRCost(Q_diag::SVector{N}, R_diag::SVector{M},
        xf::SVector{N}, uf::SVector{M}=(@SVector zeros(M))) where {N,M}
    Q = Diagonal(Q_diag)
    R = Diagonal(R_diag)
    q = -Q*xf
    r = -R*uf
    c = 0.5*(xf'Q*xf + uf'R*uf)
    return QuadraticCost(Q,R,q,r,c)
end
