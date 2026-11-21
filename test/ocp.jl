# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestOCP

using Test
using JuMP
import Fatrop
import Ipopt

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

_set_stage(x, k) = set_attribute(x, Fatrop.Stage(), k)

_ocp(model) = unsafe_backend(model).ocp

# Example of the C interface tests of Fatrop:
#   min  u0^2 + (x1 - 3)^2   s.t.  x1 = x0 + u0,  x0 = 1
# with optimum `u0 = 1`, `x1 = 2`.
function test_two_stages()
    model = Model(Fatrop.Optimizer)
    set_silent(model)
    @variable(model, x0)
    @variable(model, u0)
    @variable(model, x1)
    _set_stage(x0, 0)
    _set_stage(u0, 0)
    _set_stage(x1, 1)
    @constraint(model, c0, x0 == 1)
    # The coefficient `2` checks that the dynamics are scaled
    @constraint(model, dyn, 2x1 == 2x0 + 2u0)
    @objective(model, Min, u0^2 + (x1 - 3)^2)
    optimize!(model)
    @test termination_status(model) == LOCALLY_SOLVED
    @test primal_status(model) == FEASIBLE_POINT
    @test dual_status(model) == FEASIBLE_POINT
    @test value(x0) ≈ 1 atol = 1e-6
    @test value(u0) ≈ 1 atol = 1e-6
    @test value(x1) ≈ 2 atol = 1e-6
    @test objective_value(model) ≈ 2 atol = 1e-6
    # The Lagrangian is
    # `u0^2 + (x1 - 3)^2 - λ0 * x0 - λ * (2x1 - 2x0 - 2u0)`
    # so `2u0 + 2λ = 0` and `-λ0 + 2λ = 0`
    @test dual(dyn) ≈ -1 atol = 1e-6
    @test dual(c0) ≈ -2 atol = 1e-6
    ocp = _ocp(model)
    @test ocp.K == 2
    @test ocp.nu == [2, 0]
    @test ocp.nx == [0, 1]
    @test ocp.ng == [1, 0]
    @test ocp.ng_ineq == [0, 0]
    return
end

# Discretized double integrator with bounded control, a nonlinear objective and
# a nonlinear path constraint
function _double_integrator(optimizer; N = 20, stage = true)
    dt = 0.1
    model = Model(optimizer)
    set_silent(model)
    @variable(model, p[1:(N+1)], start = 0.5)
    @variable(model, v[1:(N+1)])
    @variable(model, -1 <= u[1:N] <= 1)
    if stage
        for k in 1:(N+1)
            _set_stage(p[k], k)
            _set_stage(v[k], k)
            if k <= N
                _set_stage(u[k], k)
            end
        end
    end
    @constraint(model, p[1] == 0)
    @constraint(model, v[1] == 0)
    @constraint(model, [k in 1:N], p[k+1] == p[k] + dt * v[k])
    @constraint(model, [k in 1:N], v[k+1] == v[k] + dt * sin(u[k]))
    @constraint(model, [k in 1:(N+1)], p[k]^2 + v[k]^2 <= 4)
    @constraint(model, p[N+1] == 0.5)
    @constraint(model, v[N+1] == 0)
    @objective(
        model,
        Min,
        sum(u[k]^2 for k in 1:N) + sum(exp(v[k]) for k in 1:(N+1))
    )
    optimize!(model)
    return model
end

function test_double_integrator()
    N = 20
    model = _double_integrator(Fatrop.Optimizer; N)
    @test termination_status(model) == LOCALLY_SOLVED
    ocp = _ocp(model)
    @test ocp.K == N + 1
    # The initial state is a control of the first stage
    @test ocp.nu == [3; ones(Int, N - 1); 0]
    @test ocp.nx == [0; fill(2, N)]
    @test ocp.ng == [2; zeros(Int, N - 1); 2]
    # The bounds on `u` and the constraint on `p[k]^2 + v[k]^2`
    @test ocp.ng_ineq == [fill(3, N); 1]
    ipopt = _double_integrator(Ipopt.Optimizer; N, stage = false)
    @test termination_status(ipopt) == LOCALLY_SOLVED
    @test objective_value(model) ≈ objective_value(ipopt) rtol = 1e-6
    for (x, y) in zip(all_variables(model), all_variables(ipopt))
        @test value(x) ≈ value(y) atol = 1e-5
    end
    for (F, S) in list_of_constraint_types(model)
        for (c, d) in
            zip(all_constraints(model, F, S), all_constraints(ipopt, F, S))
            @test value(c) ≈ value(d) atol = 1e-5
            @test dual(c) ≈ dual(d) atol = 1e-4
        end
    end
    return
end

function test_without_stage()
    model = _double_integrator(Fatrop.Optimizer; stage = false)
    @test termination_status(model) == LOCALLY_SOLVED
    @test _ocp(model).K == 1
    staged = _double_integrator(Fatrop.Optimizer)
    @test objective_value(model) ≈ objective_value(staged) rtol = 1e-6
    return
end

function _maximize(optimizer, stage)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, x <= 1)
    @variable(model, y)
    if stage
        _set_stage(x, 1)
        _set_stage(y, 2)
    end
    @constraint(model, c, 2y == x)
    @objective(model, Max, y - (x - 2)^2)
    optimize!(model)
    return model
end

function test_maximize()
    model = _maximize(Fatrop.Optimizer, true)
    ipopt = _maximize(Ipopt.Optimizer, false)
    @test termination_status(model) == LOCALLY_SOLVED
    @test value(model[:x]) ≈ 1 atol = 1e-6
    @test value(model[:y]) ≈ 0.5 atol = 1e-6
    @test objective_value(model) ≈ -0.5 atol = 1e-6
    @test dual(model[:c]) ≈ dual(ipopt[:c]) atol = 1e-6
    @test dual(UpperBoundRef(model[:x])) ≈ dual(UpperBoundRef(ipopt[:x])) atol =
        1e-6
    @test !iszero(dual(UpperBoundRef(model[:x])))
    return
end

function test_max_iter()
    model = Model(Fatrop.Optimizer)
    set_silent(model)
    set_attribute(model, "max_iter", 1)
    @variable(model, x, start = 1)
    @objective(model, Min, (x - 2)^4)
    optimize!(model)
    @test termination_status(model) == ITERATION_LIMIT
    @test primal_status(model) == MOI.UNKNOWN_RESULT_STATUS
    @test dual_status(model) == MOI.UNKNOWN_RESULT_STATUS
    @test raw_status(model) == "MaxIterExceeded"
    @test barrier_iterations(model) > 0
    @test solve_time(model) > 0
    return
end

function test_unknown_option()
    model = Model(Fatrop.Optimizer)
    @test_throws MOI.UnsupportedAttribute set_attribute(model, "unknown", 1)
    set_attribute(model, "tol", 1)
    @test get_attribute(model, "tol") === 1.0
    return
end

function test_partial_stage()
    model = Model(Fatrop.Optimizer)
    @variable(model, x)
    @variable(model, y)
    _set_stage(x, 1)
    @test_throws ErrorException(
        "The `Fatrop.Stage` attribute is set for some variables but not for " *
        "the variable $(index(y)).",
    ) optimize!(model)
    return
end

function test_non_consecutive_stages()
    model = Model(Fatrop.Optimizer)
    @variable(model, x[1:3])
    for k in 1:3
        _set_stage(x[k], k)
    end
    @constraint(model, c, x[3] == x[1])
    err = try
        optimize!(model)
    catch err
        err
    end
    @test occursin("involves variables of the stages [1, 3]", err.msg)
    return
end

function test_two_variables_of_next_stage()
    model = Model(Fatrop.Optimizer)
    @variable(model, x[1:3])
    _set_stage(x[1], 1)
    _set_stage(x[2], 2)
    _set_stage(x[3], 2)
    @constraint(model, x[2] + x[3] == x[1])
    @test_throws ErrorException optimize!(model)
    return
end

function test_nonlinear_in_next_state()
    model = Model(Fatrop.Optimizer)
    @variable(model, x[1:2])
    _set_stage(x[1], 1)
    _set_stage(x[2], 2)
    @constraint(model, c, x[2]^2 == x[1])
    @test_throws ErrorException(
        "The dynamics constraint $(index(c)) is not affine in the variable " *
        "$(index(x[2])) of the next stage.",
    ) optimize!(model)
    return
end

function test_same_state_twice()
    model = Model(Fatrop.Optimizer)
    @variable(model, x[1:2])
    _set_stage(x[1], 1)
    _set_stage(x[2], 2)
    @constraint(model, x[2] == x[1])
    @constraint(model, x[2] == 2x[1])
    @test_throws ErrorException(
        "Several dynamics constraints define the same state of stage 2.",
    ) optimize!(model)
    return
end

function test_hessian_coupling_stages()
    model = Model(Fatrop.Optimizer)
    @variable(model, x[1:2])
    _set_stage(x[1], 1)
    _set_stage(x[2], 2)
    @objective(model, Min, x[1] * x[2])
    err = try
        optimize!(model)
    catch err
        err
    end
    @test occursin("Only the dynamics constraints may couple", err.msg)
    return
end

function test_user_defined_operator()
    model = Model(Fatrop.Optimizer)
    set_silent(model)
    @variable(model, x, start = 1)
    @operator(model, op_f, 1, x -> (x - 2)^2)
    @objective(model, Min, op_f(x))
    optimize!(model)
    @test termination_status(model) == LOCALLY_SOLVED
    @test value(x) ≈ 2 atol = 1e-4
    @test :op_f in get_attribute(model, MOI.ListOfSupportedNonlinearOperators())
    return
end

function test_user_defined_operator_without_hessian()
    model = Model(Fatrop.Optimizer)
    @variable(model, x[1:2])
    f(x, y) = (x - 1)^2 + (y - 2)^2
    function ∇f(g, x, y)
        g[1] = 2 * (x - 1)
        g[2] = 2 * (y - 2)
        return
    end
    @operator(model, op_f, 2, f, ∇f)
    @objective(model, Min, op_f(x...))
    @test_throws ErrorException(
        "Fatrop requires the Hessian of the Lagrangian. Provide the Hessian " *
        "of multivariate user-defined operators.",
    ) optimize!(model)
    return
end

function test_callback_exception()
    model = Model(Fatrop.Optimizer)
    set_silent(model)
    @variable(model, x, start = 1)
    @operator(model, op_f, 1, x -> x > 1.5 ? error("boom") : (x - 2)^2)
    @objective(model, Min, op_f(x))
    err = try
        optimize!(model)
    catch err
        err
    end
    @test err isa CapturedException
    @test err.ex == ErrorException("boom")
    return
end

function test_modify_after_optimize()
    model = Model(Fatrop.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    @constraint(model, x[2] == x[1])
    @objective(model, Min, (x[1] - 1)^2 + x[2]^2)
    optimize!(model)
    @test _ocp(model).K == 1
    # The optimizer is attached so these are set to it directly
    _set_stage(x[1], 1)
    _set_stage(x[2], 2)
    set_start_value(x[1], 1)
    @operator(model, op_f, 1, x -> x^2)
    @objective(model, Min, op_f(x[1] - 1) + x[2]^2)
    optimize!(model)
    @test _ocp(model).K == 2
    @test value(x[1]) ≈ 0.5 atol = 1e-6
    return
end

function test_status_before_optimize()
    model = Model(Fatrop.Optimizer)
    @test termination_status(model) == OPTIMIZE_NOT_CALLED
    @test MOI.get(Fatrop.Optimizer(), MOI.RawStatusString()) ==
          "Optimize not called"
    @test primal_status(model) == NO_SOLUTION
    @test_throws MOI.GetAttributeNotAllowed get_attribute(model, "tol")
    return
end

function test_free_inequality()
    model = Model(Fatrop.Optimizer)
    set_silent(model)
    @variable(model, x)
    @constraint(model, c, 2x in MOI.Interval(-Inf, Inf))
    @objective(model, Min, (x - 1)^2)
    optimize!(model)
    @test termination_status(model) == LOCALLY_SOLVED
    @test value(c) ≈ 2 atol = 1e-6
    @test dual(c) == 0
    return
end

function test_unsupported_constraint()
    model = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(model, 2)
    MOI.add_constraint(model, MOI.VectorOfVariables(x), MOI.Nonnegatives(2))
    @test_throws(
        MOI.UnsupportedConstraint{MOI.VectorOfVariables,MOI.Nonnegatives},
        MOI.copy_to(Fatrop.Optimizer(), model),
    )
    return
end

function test_linear_coefficient()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    f(head, args...) = MOI.ScalarNonlinearFunction(head, Any[args...])
    @test Fatrop._linear_coefficient(x, y) == 0
    @test Fatrop._linear_coefficient(y, y) == 1
    @test Fatrop._linear_coefficient(1.0, y) == 0
    @test Fatrop._linear_coefficient(2.0 * y + 3.0 * x - y, y) == 1
    @test Fatrop._linear_coefficient(1.0 * x * x + 2.0 * y, y) == 2
    @test isnothing(Fatrop._linear_coefficient(1.0 * x * y, y))
    @test Fatrop._linear_coefficient(f(:sin, x), y) == 0
    @test isnothing(Fatrop._linear_coefficient(f(:sin, y), y))
    @test Fatrop._linear_coefficient(f(:+, y, f(:sin, x)), y) == 1
    @test Fatrop._linear_coefficient(f(:-, y), y) == -1
    @test Fatrop._linear_coefficient(f(:-, f(:sin, x), y, y), y) == -2
    @test Fatrop._linear_coefficient(f(:*, 2.0, y, 3.0), y) == 6
    @test isnothing(Fatrop._linear_coefficient(f(:*, x, y), y))
    @test isnothing(Fatrop._linear_coefficient(f(:-, f(:sin, y)), y))
    return
end

end  # module TestOCP

TestOCP.runtests()
