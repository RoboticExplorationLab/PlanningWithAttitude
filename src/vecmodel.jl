struct VecModel{L} <: AbstractModel
    model::L
    VecModel(model::L) where L <: LieGroupModel = new{L}(model)
end

RobotDynamics.state_dim(model::VecModel) = state_dim(model.model)
RobotDynamics.control_dim(model::VecModel) = control_dim(model.model)
RobotDynamics.dynamics(model::VecModel, args...) = dynamics(model.model, args...) 