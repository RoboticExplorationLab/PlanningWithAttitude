export
	solve!,
	rollout!

############################################################################################
#                                   SOLVE
############################################################################################
function solve!(solver::iLQRSolver)
    set_verbosity!(solver.opts)
    clear_cache!(solver.opts)

    solver.stats.iterations = 0
    solver.ρ[1] = solver.opts.bp_reg_initial
    solver.dρ[1] = 0.0
    Z = solver.Z; Z̄ = solver.Z̄;

    n,m,N = size(solver)
    J = Inf
    _J = get_J(solver.obj)

    # Initial rollout
    rollout!(solver)

    cost!(solver.obj, Z)
    J_prev = sum(_J)


    for i = 1:solver.opts.iterations
		ΔV = backwardpass!(solver)
		J = forwardpass!(solver, ΔV, J_prev)

        # check for cost blow up
        if J > solver.opts.max_cost_value
            # @warn "Cost exceeded maximum cost"
            return solver
        end

        for k = 1:N
            Z[k].z = Z̄[k].z
        end

        dJ = abs(J - J_prev)
        J_prev = copy(J)
        # gradient_todorov!(solver)

        record_iteration!(solver, J, dJ)
        evaluate_convergence(solver) ? break : nothing
    end
    return solver
end

"""
Stash iteration statistics
"""
function record_iteration!(solver::iLQRSolver, J, dJ)
    solver.stats.iterations += 1
    i = solver.stats.iterations::Int
    solver.stats.cost[i] = J
    solver.stats.dJ[i] = dJ
    # solver.stats.gradient[i] = mean(solver.grad)

    @logmsg InnerLoop :iter value=i
    @logmsg InnerLoop :cost value=J
    @logmsg InnerLoop :dJ   value=dJ
    @logmsg InnerLoop :grad value=solver.stats.gradient[i]
    # @logmsg InnerLoop :zero_count value=solver.stats[:dJ_zero_counter][end]
    if solver.opts.verbose
        print_level(InnerLoop)
    end
    return nothing
end

"""
    Calculate the problem gradient using heuristic from iLQG (Todorov) solver
"""
function gradient_todorov!(solver::iLQRSolver)
    for k in eachindex(solver.d)
        solver.grad[k] = maximum( abs.(solver.d[k]) ./ (abs.(control(solver.Z[k])) .+ 1) )
    end
end

"""
Check convergence conditions for iLQR
"""
function evaluate_convergence(solver::iLQRSolver)
    # Get current iterations
    i = solver.stats.iterations

    # Check for cost convergence
    # note the  dJ > 0 criteria exists to prevent loop exit when forward pass makes no improvement
    if 0.0 < solver.stats.dJ[i] < solver.opts.cost_tolerance
        return true
    end

    # Check for gradient convergence
    # if solver.stats.gradient[i] < solver.opts.gradient_norm_tolerance
    #     return true
    # end

    # Check total iterations
    if i >= solver.opts.iterations
        return true
    end

    # Outer loop update if forward pass is repeatedly unsuccessful
    if solver.stats.dJ_zero_counter > solver.opts.dJ_counter_limit
        return true
    end

    return false
end

############################################################################################
#                                 EXPANSIONS
############################################################################################

function dynamics_jacobian(solver::iLQRSolver, z::KnotPoint{T,N,M,NM}) where {T,N,M,NM}
	model = solver.model
	∇f = discrete_jacobian(model, z)
	ix,iu = z._x, z._u
	x,u,dt = state(z), control(z), z.dt
	G1 = state_diff_jacobian(model, x)
	G2 = state_diff_jacobian(model, discrete_dynamics(model, x, u, dt))
	A = G2'∇f[ix,ix]*G1
	B = G2'∇f[ix,iu]
	return A,B
end

function cost_gradient(solver::iLQRSolver, cost::CostFunction, z::KnotPoint)
	Qx,Qu = gradient(cost, z)
	G = state_diff_jacobian(solver.model, state(z))
	return G'Qx, Qu
end

function cost_hessian(solver::iLQRSolver, cost::CostFunction, z::KnotPoint)
	Qxx,Quu,Qux = hessian(cost, z)
	G = state_diff_jacobian(solver.model, state(z))
	return G'Qxx*G, Quu, Qux*G
end

############################################################################################
#                                 iLQR METHODS
############################################################################################

@inline rollout!(solver::iLQRSolver) = rollout!(solver.model, solver.Z, solver.x0)

function rollout!(model::AbstractModel, Z::Traj, x0)
    Z[1].z = [x0; control(Z[1])]
    for k = 1:length(Z)-1
		z = Z[k]
    	x = discrete_dynamics(model, state(z), control(z), z.dt)
		set_state!(Z[k+1], x)
    end
end

function rollout!(solver::iLQRSolver, α)
    Z = solver.Z; Z̄ = solver.Z̄
    K = solver.K; d = solver.d;

    Z̄[1].z = [solver.x0; control(Z[1])]

    temp = 0.0

    for k = 1:solver.N-1
        δx = state_diff(solver.model, state(Z̄[k]), state(Z[k]))
        ū = control(Z[k]) + K[k]*δx + α*d[k]
        set_control!(Z̄[k], ū)

        # Z̄[k].z = [state(Z̄[k]); control(Z[k]) + δu]
        Z̄[k+1].z = [discrete_dynamics(solver.model, Z̄[k]); control(Z[k+1])]


        temp = norm(Z̄[k+1].z)
        if temp > solver.opts.max_state_value
            return false
        end
    end
    return true
end

function backwardpass!(solver::iLQRSolver)
    n,m,N = size(solver)

    # Objective
    obj = solver.obj
    model = solver.model

    # Extract variables
    Z = solver.Z; K = solver.K; d = solver.d;

    # Terminal cost-to-go
    Sxx = cost_hessian(solver, obj.costs[N], Z[N])[1]
    Sx = cost_gradient(solver, obj.costs[N], Z[N])[1]

    # Initialize expecte change in cost-to-go
    ΔV = @SVector zeros(2)

    k = N-1
    while k > 0
        ix = Z[k]._x
        iu = Z[k]._u

		fdx,fdu = dynamics_jacobian(solver, Z[k])
		Qx,Qu = cost_gradient(solver, obj.costs[k], Z[k])
		Qxx,Quu,Qux = cost_hessian(solver, obj.costs[k], Z[k])

		Qx += fdx'Sx
		Qu += fdu'Sx
		Qxx += fdx'Sxx*fdx
		Quu += fdu'Sxx*fdu
		Qux += fdu'Sxx*fdx

        if solver.opts.bp_reg_type == :state
            Quu_reg = Quu + solver.ρ[1]*fdu'fdu
            Qux_reg = Qux + solver.ρ[1]*fdu'fdx
        elseif solver.opts.bp_reg_type == :control
            Quu_reg = Quu + solver.ρ[1]*I
            Qux_reg = Qux
        end

        # Regularization

        # Compute gains
        K[k] = -(Quu_reg\Qux_reg)
        d[k] = -(Quu_reg\Qu)

        # Calculate cost-to-go (using unregularized Quu and Qux)
        Sx  =  Qx + K[k]'*Quu*d[k] + K[k]'* Qu + Qux'd[k]
        Sxx = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'K[k]
        Sxx = 0.5*(Sxx + Sxx')

        # calculated change is cost-to-go over entire trajectory
        ΔV += @SVector [d[k]'*Qu, 0.5*d[k]'*Quu*d[k]]

        k -= 1
    end

    regularization_update!(solver, :decrease)

    return ΔV
end

"""
Simulate the system forward using the optimal feedback gains from the backward pass,
projecting the system on the dynamically feasible subspace. Performs a line search to ensure
adequate progress on the nonlinear problem.
"""
function forwardpass!(solver::iLQRSolver, ΔV, J_prev)
    Z = solver.Z; Z̄ = solver.Z̄
    obj = solver.obj

    _J = get_J(obj)
    J::Float64 = Inf
    α = 1.0
    iter = 0
    z = -1.0
    expected = 0.0
    flag = true

    while (z ≤ solver.opts.line_search_lower_bound || z > solver.opts.line_search_upper_bound) && J >= J_prev

        # Check that maximum number of line search decrements has not occured
        if iter > solver.opts.iterations_linesearch
            for k in eachindex(Z)
                Z̄[k].z = Z[k].z
            end
            cost!(obj, Z̄)
            J = sum(_J)

            z = 0
            α = 0.0
            expected = 0.0

			@logmsg InnerLoop "Linesearch max iter(reg up)"
            regularization_update!(solver, :increase)
            solver.ρ[1] += solver.opts.bp_reg_fp
            break
        end


        # Otherwise, rollout a new trajectory for current alpha
        flag = rollout!(solver, α)

        # Check if rollout completed
        if ~flag
            # Reduce step size if rollout returns non-finite values (NaN or Inf)
            # @logmsg InnerIters "Non-finite values in rollout"
            iter += 1
            α /= 2.0
            continue
        end

        # Calcuate cost
        cost!(obj, Z̄)
        J = sum(_J)

        expected::Float64 = -α*(ΔV[1] + α*ΔV[2])
        if expected > 0.0
            z::Float64  = (J_prev - J)/expected
        else
            z = -1.0
        end

        iter += 1
        α /= 2.0
    end

    if J > J_prev
        error("Error: Cost increased during Forward Pass")
    end

    @logmsg InnerLoop :expected value=expected
    @logmsg InnerLoop :z value=z
    @logmsg InnerLoop :α value=2*α
    @logmsg InnerLoop :ρ value=solver.ρ[1]

    return J
end

"""
Update the regularzation for the iLQR backward pass
"""
function regularization_update!(solver::iLQRSolver,status::Symbol=:increase)
    # println("reg $(status)")
    if status == :increase # increase regularization
        # @logmsg InnerLoop "Regularization Increased"
        solver.dρ[1] = max(solver.dρ[1]*solver.opts.bp_reg_increase_factor, solver.opts.bp_reg_increase_factor)
        solver.ρ[1] = max(solver.ρ[1]*solver.dρ[1], solver.opts.bp_reg_min)
        # if solver.ρ[1] > solver.opts.bp_reg_max
        #     @warn "Max regularization exceeded"
        # end
    elseif status == :decrease # decrease regularization
        # TODO: Avoid divides by storing the decrease factor (divides are 10x slower)
        solver.dρ[1] = min(solver.dρ[1]/solver.opts.bp_reg_increase_factor, 1.0/solver.opts.bp_reg_increase_factor)
        solver.ρ[1] = solver.ρ[1]*solver.dρ[1]*(solver.ρ[1]*solver.dρ[1]>solver.opts.bp_reg_min)
    end
end
