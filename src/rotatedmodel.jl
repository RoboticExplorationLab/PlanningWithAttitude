struct QuatSlackModel{R,M,L} <: RD.RigidBody{R} 
    model::L
    function QuatSlackModel(model::L) where L <: RD.RigidBody
        M = RD.control_dim(model)
        R = RD.rotation_type(model)
        new{R,M,L}(model)
    end
end
RobotDynamics.control_dim(::QuatSlackModel{<:Any,M}) where M = M+1 

function RobotDynamics.dynamics(model::QuatSlackModel{<:Any,m}, x, u, t=0) where m
    uinds = SVector{m}(1:m)
    u0 = u[uinds]                    # original controls
    s = u[m+1]                       # Quaternion slack
    r,q,v,ω = RobotDynamics.parse_state(model, x)
    q̄ = s*q                          # scale the quaternion by the extra control
    x̄ = RobotDynamics.build_state(model, r, q̄, v, ω)
    RD.dynamics(model.model, x̄, u, t)   # call the original dynamics with the scaled quaternion
end

"""
    UnitQuatConstraint

Enforce a that the unit quaternion in the state, at indices `qind`, has unit norm
after being multiplied by a slack variable at index `sind` in the joint state-control vector
`z = [x;u]`.
"""
struct UnitQuatConstraint <: TrajectoryOptimization.StageConstraint
    n::Int
    m::Int
    qind::SVector{4,Int}
    sind::Int
end

function UnitQuatConstraint(n::Int, m::Int, qind=4:7, sind=n+m)
    UnitQuatConstraint(n, m, SVector{4}(qind), sind)
end

UnitQuatConstraint(model::QuatSlackModel) = UnitQuatConstraint(size(model)...)

Base.copy(con::UnitQuatConstraint) = UnitQuatConstraint(con.n, con.m, con.qind, con.sind)

@inline TO.sense(::UnitQuatConstraint) = Equality()
@inline Base.length(::UnitQuatConstraint) = 1
@inline RobotDynamics.state_dim(con::UnitQuatConstraint) = con.n
@inline RobotDynamics.control_dim(con::UnitQuatConstraint) = con.m

function TO.evaluate(con::UnitQuatConstraint, z::RD.AbstractKnotPoint) 
    q = z.z[con.qind]
    s = z.z[con.sind]
    q̄ = q*s
    return SA[dot(q̄,q̄) - 1]
end

function TO.jacobian!(∇c, con::UnitQuatConstraint, z::RD.AbstractKnotPoint)
    q = z.z[con.qind]
    s = z.z[con.sind]
    q̄ = q*s
    for (i,j) in enumerate(con.qind)
        ∇c[1,j] = 2*q̄[i]*s
    end
    ∇c[con.sind] = 2s*dot(q,q)
    return false
end

function TO.change_dimension(con::UnitQuatConstraint, n::Int, m::Int, xi=1:n, ui=1:m)
    zi = [xi; ui]
    UnitQuatConstraint(n, m, zi[con.qind], zi[con.sind])
end