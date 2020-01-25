export
    iLQRSolverOptions,
    iLQRSolver,
    initial_controls!,
    states,
    controls

@with_kw mutable struct iLQRStats{T}
    iterations::Int = 0
    cost::Vector{T} = [0.]
    dJ::Vector{T} = [0.]
    gradient::Vector{T} = [0.]
    dJ_zero_counter::Int = 0
end

function reset!(stats::iLQRStats, N=0)
    stats.iterations = 0
    stats.cost = zeros(N)
    stats.dJ = zeros(N)
    stats.gradient = zeros(N)
    stats.dJ_zero_counter = 0
end


@with_kw mutable struct iLQRSolverOptions{T}
    # Options

    "Print summary at each iteration."
    verbose::Bool=false

    "Live plotting."
    live_plotting::Symbol=:off # :state, :control

    "dJ < ϵ, cost convergence criteria for unconstrained solve or to enter outerloop for constrained solve."
    cost_tolerance::T = 1.0e-4

    "gradient type: :todorov, :feedforward."
    gradient_type::Symbol = :todorov

    "gradient_norm < ϵ, gradient norm convergence criteria."
    gradient_norm_tolerance::T = 1.0e-5

    "iLQR iterations."
    iterations::Int = 300

    "restricts the total number of times a forward pass fails, resulting in regularization, before exiting."
    dJ_counter_limit::Int = 10

    "use square root method backward pass for numerical conditioning."
    square_root::Bool = false

    "forward pass approximate line search lower bound, 0 < line_search_lower_bound < line_search_upper_bound."
    line_search_lower_bound::T = 1.0e-8

    "forward pass approximate line search upper bound, 0 < line_search_lower_bound < line_search_upper_bound < ∞."
    line_search_upper_bound::T = 10.0

    "maximum number of backtracking steps during forward pass line search."
    iterations_linesearch::Int = 20

    # Regularization
    "initial regularization."
    bp_reg_initial::T = 0.0

    "regularization scaling factor."
    bp_reg_increase_factor::T = 1.6

    "maximum regularization value."
    bp_reg_max::T = 1.0e8

    "minimum regularization value."
    bp_reg_min::T = 1.0e-8

    "type of regularization- control: () + ρI, state: (S + ρI); see Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization."
    bp_reg_type::Symbol = :control

    "additive regularization when forward pass reaches max iterations."
    bp_reg_fp::T = 10.0

    # square root backward pass options:
    "type of matrix inversion for bp sqrt step."
    bp_sqrt_inv_type::Symbol = :pseudo

    "initial regularization for square root method."
    bp_reg_sqrt_initial::T = 1.0e-6

    "regularization scaling factor for square root method."
    bp_reg_sqrt_increase_factor::T = 10.0

    # Solver Numerical Limits
    "maximum cost value, if exceded solve will error."
    max_cost_value::T = 1.0e8

    "maximum state value, evaluated during rollout, if exceded solve will error."
    max_state_value::T = 1.0e8

    "maximum control value, evaluated during rollout, if exceded solve will error."
    max_control_value::T = 1.0e8

    log_level::Base.CoreLogging.LogLevel = InnerLoop
end

struct iLQRSolver{L,C,n,m,nm}
    # Model
    model::L
    obj::Objective{C}

    # Problem Info
    x0::SVector{n,Float64}
    tf::Float64
    N::Int

    opts::iLQRSolverOptions{Float64}
    stats::iLQRStats{Float64}

    # Trajectories
    Z::Vector{KnotPoint{Float64,n,m,nm}}
    Z̄::Vector{KnotPoint{Float64,n,m,nm}}

    # Data
    K::Vector{Matrix{Float64}}
    d::Vector{SVector{m,Float64}}

    ρ::Vector{Float64}
    dρ::Vector{Float64}

    logger::SolverLogger
end

function iLQRSolver(model::L, obj::Objective, x0, tf::Real, N::Int;
        opts=iLQRSolverOptions{Float64}()) where L
    n,m = size(model)
    dt = tf/(N-1)
    stats = iLQRStats()
    reset!(stats, opts.iterations)
    Z = Traj(n,m,dt,N)
    Z̄ = Traj(n,m,dt,N)
    K = [zeros(m,n) for k = 1:N-1]
    d = [@SVector zeros(m) for k = 1:N-1]
    ρ  = [0.]
    dρ = [0.]
    logger = default_logger(opts.verbose)
    return iLQRSolver(model, obj, x0, tf, N, opts, stats, Z, Z̄, K, d, ρ, dρ, logger)
end

Base.size(solver::iLQRSolver{L,C,n,m}) where {L,C,n,m} = n,m,solver.N

function initial_controls!(solver::iLQRSolver, U0)
    set_controls!(solver.Z, U0)
end

@inline states(solver::iLQRSolver) = state.(solver.Z)
@inline controls(solver::iLQRSolver) = control.(solver.Z)
