## Quaternion Goal Constraints
struct QuatGeoCon{T} <: TO.StateConstraint
    n::Int
    qf::UnitQuaternion{T}
    qind::SVector{4,Int}
end
function TO.evaluate(con::QuatGeoCon, x::SVector)
    q = x[con.qind]
    qf = Rotations.params(con.qf)
    dq = qf'q
    return SA[min(1-dq, 1+dq)]
end
TO.sense(::QuatGeoCon) = TO.Equality()
TO.state_dim(con::QuatGeoCon) = con.n
Base.length(::QuatGeoCon) = 1


struct QuatErr{T} <: TO.StateConstraint
    n::Int
    qf::UnitQuaternion{T}
    qind::SVector{4,Int}
end
function TO.evaluate(con::QuatErr, x::StaticVector)
    qf = con.qf
    q = UnitQuaternion(x[con.qind])
    return Rotations.rotation_error(q, qf, Rotations.MRPMap()) 
end
TO.sense(::QuatErr) = TO.Equality()
TO.state_dim(con::QuatErr) = con.n
Base.length(con::QuatErr) = 3

struct QuatVecEq{T} <: TO.StateConstraint
    n::Int
    qf::UnitQuaternion{T}
    qind::SVector{4,Int}
end
function TO.evaluate(con::QuatVecEq, x::StaticVector)
    qf = Rotations.params(con.qf)
    q = normalize(x[con.qind])
    dq = qf'q
    if dq < 0
        qf *= -1
    end
    return SA[qf[1] - q[1], qf[2] - q[2], qf[3] - q[3], qf[4] - q[4]] 
end
TO.sense(::QuatVecEq) = TO.Equality()
TO.state_dim(con::QuatVecEq) = con.n
Base.length(con::QuatVecEq) = 4


