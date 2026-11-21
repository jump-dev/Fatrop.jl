# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    Stage() <: MOI.AbstractVariableAttribute

The stage of a variable in the optimal control problem. The value is an
`Integer`, the stages of the problem are the distinct values of this attribute,
sorted in increasing order.

The variables of a stage are not explicitly classified into states and
controls. Instead, an equality constraint involving variables of two
consecutive stages `k` and `k + 1` is detected as a dynamics constraint
`x[k+1][i] = f(u[k], x[k])`. For this, it must involve exactly one variable of
stage `k + 1`, which must appear affinely. The variables of stage `k + 1`
defined by such constraints are its states. All other variables are controls.

All other constraints, as well as the Hessian of the objective, must only
involve variables of a single stage.

If this attribute is not set for any variable, all the variables are in the
same stage so any nonlinear program can be solved but no structure is
exploited.
"""
struct Stage <: MOI.AbstractVariableAttribute end

@enum(_ConstraintKind, _PATH, _DYNAMICS, _INEQUALITY)

const _SCALAR_FUNCTIONS = Union{
    MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
}

const _SCALAR_SETS = Union{
    MOI.EqualTo{Float64},
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.Interval{Float64},
}

# A scalar constraint of the model. In Fatrop, its residual is
# `scale * (g(x) - constant)` where `g` is the constraint function.
struct _Row
    ci::MOI.ConstraintIndex
    kind::_ConstraintKind
    # 1-based stage of the constraint. For dynamics, it is the stage `k` of
    # `x[k+1] = f(u[k], x[k])`.
    stage::Int
    # 1-based index of the constraint in its block of the stage
    index::Int
    scale::Float64
    constant::Float64
    lower::Float64
    upper::Float64
end

_bounds(set::MOI.GreaterThan) = set.lower, Inf
_bounds(set::MOI.LessThan) = -Inf, set.upper
_bounds(set::MOI.Interval) = set.lower, set.upper

_add_variables!(vars, f::MOI.VariableIndex) = push!(vars, f)
_add_variables!(_, ::Real) = nothing

function _add_variables!(vars, f::MOI.ScalarAffineFunction)
    for term in f.terms
        push!(vars, term.variable)
    end
    return
end

function _add_variables!(vars, f::MOI.ScalarQuadraticFunction)
    for term in f.quadratic_terms
        push!(vars, term.variable_1)
        push!(vars, term.variable_2)
    end
    for term in f.affine_terms
        push!(vars, term.variable)
    end
    return
end

function _add_variables!(vars, f::MOI.ScalarNonlinearFunction)
    for arg in f.args
        _add_variables!(vars, arg)
    end
    return
end

function _variables(f)
    vars = Set{MOI.VariableIndex}()
    _add_variables!(vars, f)
    return vars
end

_contains(f, y::MOI.VariableIndex) = y in _variables(f)

# Return the coefficient `a` such that `f(x) = a * y + h(x)` where `h` does not
# depend on `y`. Return `nothing` if `f` is not affine in `y` or if this is not
# apparent from the expression tree.
_linear_coefficient(f::MOI.VariableIndex, y) = f == y ? 1.0 : 0.0
_linear_coefficient(::Real, _) = 0.0

function _linear_coefficient(f::MOI.ScalarAffineFunction, y)
    return sum(t.coefficient for t in f.terms if t.variable == y; init = 0.0)
end

function _linear_coefficient(f::MOI.ScalarQuadraticFunction, y)
    for term in f.quadratic_terms
        if term.variable_1 == y || term.variable_2 == y
            return
        end
    end
    return sum(
        t.coefficient for t in f.affine_terms if t.variable == y;
        init = 0.0,
    )
end

function _linear_coefficient(f::MOI.ScalarNonlinearFunction, y)
    if !_contains(f, y)
        return 0.0
    end
    coefs = [_linear_coefficient(arg, y) for arg in f.args]
    if any(isnothing, coefs)
        return
    end
    if f.head == :+
        return sum(coefs)
    elseif f.head == :-
        if length(coefs) == 1
            return -coefs[1]
        end
        return coefs[1] - sum(coefs[2:end])
    elseif f.head == :*
        # `y` is multiplied by constants only
        if count(!iszero, coefs) == 1 &&
           count(arg -> !(arg isa Real), f.args) == 1
            return prod(arg for arg in f.args if arg isa Real; init = 1.0) *
                   sum(coefs)
        end
    end
    return
end

# Fatrop fails on an inequality constraint with infinite lower and upper bounds
# and on an equality constraint with a zero Jacobian so we do not give these
# constraints to Fatrop.
function _is_trivially_satisfied(model, f, set)
    if !(set isa MOI.EqualTo) && all(isinf, _bounds(set))
        return true
    end
    if isempty(_variables(f))
        value = MOI.Utilities.eval_variables(_ -> 0.0, model, f)
        return iszero(MOI.Utilities.distance_to_set(value, set))
    end
    return false
end

function _register_operators(nlp::MOI.Nonlinear.Model, model::MOI.ModelLike)
    for attr in MOI.get(model, MOI.ListOfModelAttributesSet())
        if attr isa MOI.UserDefinedFunction
            f = MOI.get(model, attr)
            MOI.Nonlinear.register_operator(nlp, attr.name, attr.arity, f...)
        end
    end
    return
end

function _stages(model::MOI.ModelLike, vars::Vector{MOI.VariableIndex})
    values = Any[MOI.get(model, Stage(), vi) for vi in vars]
    if all(isnothing, values)
        return ones(Int, length(vars)), 1
    end
    for (vi, value) in zip(vars, values)
        if isnothing(value)
            error(
                "The `Fatrop.Stage` attribute is set for some variables but " *
                "not for the variable $vi.",
            )
        end
    end
    sorted = sort!(unique(values))
    position = Dict(value => k for (k, value) in enumerate(sorted))
    return Int[position[value] for value in values], length(sorted)
end

"""
    struct _OCP

Optimal control problem structure detected from the MOI model together with an
evaluator of its derivatives.

The variables of the evaluator are ordered as Fatrop's primal vector
`[u[1]; x[1]; u[2]; x[2]; ...]` and its constraints are ordered as Fatrop's
equality constraints: the path equality constraints of all stages, then the
dynamics, then the inequality constraints.
"""
struct _OCP
    K::Int
    nu::Vector{Int}
    nx::Vector{Int}
    ng::Vector{Int}
    ng_ineq::Vector{Int}
    # `variables[i]` is the variable at index `i` of Fatrop's primal vector
    variables::Vector{MOI.VariableIndex}
    # `variables[position[vi]] == vi`
    position::Dict{MOI.VariableIndex,Int}
    # `offsets[k]` is the offset of `u[k]` in Fatrop's primal vector
    offsets::Vector{Int}
    # The stage of each entry of Fatrop's primal vector
    variable_stage::Vector{Int}
    rows::Vector{_Row}
    # `rows[ineq_offsets[k] + i]` is the `i`th inequality constraint of stage
    # `k`
    ineq_offsets::Vector{Int}
    row_of::Dict{MOI.ConstraintIndex,Int}
    has_objective::Bool
    sense::Float64
    evaluator::MOI.Nonlinear.Evaluator
    jacobian_structure::Vector{Tuple{Int,Int}}
    hessian_structure::Vector{Tuple{Int,Int}}
    # Buffers
    jacobian::Vector{Float64}
    hessian::Vector{Float64}
    multipliers::Vector{Float64}
    # One dense matrix per stage, they are packed in Fatrop's matrices
    RSQ::Vector{Matrix{Float64}}
    BA::Vector{Matrix{Float64}}
    Gg::Vector{Matrix{Float64}}
    Gg_ineq::Vector{Matrix{Float64}}
end

function _OCP(model::MOI.ModelLike)
    all_vars = MOI.get(model, MOI.ListOfVariableIndices())
    stage_values, K = _stages(model, all_vars)
    stage_of = Dict(zip(all_vars, stage_values))
    path = [_Row[] for _ in 1:K]
    ineq = [_Row[] for _ in 1:K]
    # `dynamics[k]` defines the states of stage `k`
    dynamics = [Pair{MOI.VariableIndex,_Row}[] for _ in 1:K]
    functions = Dict{MOI.ConstraintIndex,Any}()
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
            f = MOI.get(model, MOI.ConstraintFunction(), ci)
            set = MOI.get(model, MOI.ConstraintSet(), ci)
            if _is_trivially_satisfied(model, f, set)
                continue
            end
            functions[ci] = f
            # A constraint that is constant but not satisfied is kept in the
            # first stage, Fatrop will report the problem as infeasible.
            stages = sort!(unique(stage_of[vi] for vi in _variables(f)))
            k = isempty(stages) ? 1 : first(stages)
            if length(stages) <= 1
                if set isa MOI.EqualTo
                    push!(
                        path[k],
                        _Row(ci, _PATH, k, 0, 1.0, set.value, NaN, NaN),
                    )
                else
                    push!(
                        ineq[k],
                        _Row(ci, _INEQUALITY, k, 0, 1.0, 0.0, _bounds(set)...),
                    )
                end
                continue
            end
            next = [vi for vi in _variables(f) if stage_of[vi] == k + 1]
            if length(stages) != 2 ||
               stages[2] != k + 1 ||
               !(set isa MOI.EqualTo) ||
               length(next) != 1
                error(
                    "The constraint $ci involves variables of the stages " *
                    "$stages. A constraint involving several stages must be " *
                    "an equality constraint involving the variables of two " *
                    "consecutive stages `k` and `k + 1` with exactly one " *
                    "variable of stage `k + 1`.",
                )
            end
            y = only(next)
            a = _linear_coefficient(f, y)
            if isnothing(a) || iszero(a)
                error(
                    "The dynamics constraint $ci is not affine in the " *
                    "variable $y of the next stage.",
                )
            end
            # `x[k+1] = f(u[k], x[k])` is `0 = a * y + h(x) - c` so Fatrop's
            # residual `f(u[k], x[k]) - x[k+1]` is `-(a * y + h(x) - c) / a`.
            push!(
                dynamics[k+1],
                y => _Row(ci, _DYNAMICS, k, 0, -1 / a, set.value, NaN, NaN),
            )
        end
    end
    variables = MOI.VariableIndex[]
    offsets = zeros(Int, K)
    nu = zeros(Int, K)
    nx = zeros(Int, K)
    for k in 1:K
        states = first.(dynamics[k])
        if !allunique(states)
            error(
                "Several dynamics constraints define the same state of stage $k.",
            )
        end
        offsets[k] = length(variables)
        state_set = Set(states)
        for vi in all_vars
            if stage_of[vi] == k && !(vi in state_set)
                push!(variables, vi)
            end
        end
        nu[k] = length(variables) - offsets[k]
        append!(variables, states)
        nx[k] = length(states)
    end
    rows = _Row[]
    for k in 1:K, (i, row) in enumerate(path[k])
        push!(
            rows,
            _Row(
                row.ci,
                row.kind,
                row.stage,
                i,
                row.scale,
                row.constant,
                row.lower,
                row.upper,
            ),
        )
    end
    for k in 2:K, (i, (_, row)) in enumerate(dynamics[k])
        push!(
            rows,
            _Row(
                row.ci,
                row.kind,
                row.stage,
                i,
                row.scale,
                row.constant,
                row.lower,
                row.upper,
            ),
        )
    end
    ineq_offsets = zeros(Int, K)
    for k in 1:K
        ineq_offsets[k] = length(rows)
        for (i, row) in enumerate(ineq[k])
            push!(
                rows,
                _Row(
                    row.ci,
                    row.kind,
                    row.stage,
                    i,
                    row.scale,
                    row.constant,
                    row.lower,
                    row.upper,
                ),
            )
        end
    end
    nlp = MOI.Nonlinear.Model()
    _register_operators(nlp, model)
    sense = MOI.get(model, MOI.ObjectiveSense())
    has_objective = sense != MOI.FEASIBILITY_SENSE
    if has_objective
        F = MOI.get(model, MOI.ObjectiveFunctionType())
        MOI.Nonlinear.set_objective(
            nlp,
            MOI.get(model, MOI.ObjectiveFunction{F}()),
        )
    end
    for row in rows
        f = functions[row.ci]
        MOI.Nonlinear.add_constraint(nlp, f, MOI.EqualTo(0.0))
    end
    evaluator = MOI.Nonlinear.Evaluator(
        nlp,
        MOI.Nonlinear.SparseReverseMode(),
        variables,
    )
    if !(:Hess in MOI.features_available(evaluator))
        error(
            "Fatrop requires the Hessian of the Lagrangian. Provide the " *
            "Hessian of multivariate user-defined operators.",
        )
    end
    MOI.initialize(evaluator, [:Grad, :Jac, :Hess])
    jacobian_structure = MOI.jacobian_structure(evaluator)
    hessian_structure = MOI.hessian_lagrangian_structure(evaluator)
    variable_stage = Int[stage_of[vi] for vi in variables]
    for (i, j) in hessian_structure
        if variable_stage[i] != variable_stage[j]
            error(
                "The Hessian of the Lagrangian has a nonzero between the " *
                "variable $(variables[i]) of stage $(variable_stage[i]) and " *
                "the variable $(variables[j]) of stage $(variable_stage[j]). " *
                "Only the dynamics constraints may couple different stages.",
            )
        end
    end
    n = nu .+ nx
    return _OCP(
        K,
        nu,
        nx,
        [length(path[k]) for k in 1:K],
        [length(ineq[k]) for k in 1:K],
        variables,
        Dict(vi => i for (i, vi) in enumerate(variables)),
        offsets,
        variable_stage,
        rows,
        ineq_offsets,
        Dict(row.ci => r for (r, row) in enumerate(rows)),
        has_objective,
        sense == MOI.MAX_SENSE ? -1.0 : 1.0,
        evaluator,
        jacobian_structure,
        hessian_structure,
        zeros(length(jacobian_structure)),
        zeros(length(hessian_structure)),
        zeros(length(rows)),
        [zeros(n[k], n[k]) for k in 1:K],
        [zeros(n[k], k < K ? nx[k+1] : 0) for k in 1:K],
        [zeros(n[k], length(path[k])) for k in 1:K],
        [zeros(n[k], length(ineq[k])) for k in 1:K],
    )
end

# Position of the variable at index `i` of Fatrop's primal vector in
# `[u[k]; x[k]]`
_local(ocp::_OCP, i) = i - ocp.offsets[ocp.variable_stage[i]]

function _eval_objective(ocp::_OCP, x)
    if !ocp.has_objective
        return 0.0
    end
    return ocp.sense * MOI.eval_objective(ocp.evaluator, x)
end

function _eval_objective_gradient(ocp::_OCP, grad, x)
    if ocp.has_objective
        MOI.eval_objective_gradient(ocp.evaluator, grad, x)
        grad .*= ocp.sense
    else
        fill!(grad, 0.0)
    end
    return
end

function _eval_constraint_violation(ocp::_OCP, res, x)
    MOI.eval_constraint(ocp.evaluator, res, x)
    for (r, row) in enumerate(ocp.rows)
        res[r] = row.scale * (res[r] - row.constant)
    end
    return
end

function _eval_constraint_jacobian(ocp::_OCP, x)
    MOI.eval_constraint_jacobian(ocp.evaluator, ocp.jacobian, x)
    for k in 1:ocp.K
        fill!(ocp.BA[k], 0.0)
        fill!(ocp.Gg[k], 0.0)
        fill!(ocp.Gg_ineq[k], 0.0)
    end
    for ((r, i), value) in zip(ocp.jacobian_structure, ocp.jacobian)
        row = ocp.rows[r]
        k = row.stage
        if ocp.variable_stage[i] != k
            # This is the state of stage `k + 1` defined by the dynamics,
            # Fatrop assumes its coefficient is `-1`.
            continue
        end
        if row.kind == _PATH
            A = ocp.Gg[k]
        elseif row.kind == _DYNAMICS
            A = ocp.BA[k]
        else
            A = ocp.Gg_ineq[k]
        end
        # Fatrop's matrices are transposed: one column per constraint
        A[_local(ocp, i), row.index] += row.scale * value
    end
    return
end

function _eval_hessian_lagrangian(ocp::_OCP, x, objective_scale, λ)
    for (r, row) in enumerate(ocp.rows)
        ocp.multipliers[r] = row.scale * λ[r]
    end
    σ = ocp.has_objective ? ocp.sense * objective_scale : 0.0
    MOI.eval_hessian_lagrangian(
        ocp.evaluator,
        ocp.hessian,
        x,
        σ,
        ocp.multipliers,
    )
    for k in 1:ocp.K
        fill!(ocp.RSQ[k], 0.0)
    end
    for ((i, j), value) in zip(ocp.hessian_structure, ocp.hessian)
        H = ocp.RSQ[ocp.variable_stage[i]]
        li, lj = _local(ocp, i), _local(ocp, j)
        H[li, lj] += value
        if li != lj
            H[lj, li] += value
        end
    end
    return
end
