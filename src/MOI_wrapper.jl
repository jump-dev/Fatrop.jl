# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

const _MODEL = MOI.Utilities.UniversalFallback{MOI.Utilities.Model{Float64}}

"""
    Optimizer()

Create a new Fatrop optimizer. See [`Stage`](@ref) for how to communicate the
optimal control structure of the problem.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    model::_MODEL
    options::Dict{String,Any}
    silent::Bool
    # Set by `MOI.optimize!`
    ocp::Union{Nothing,_OCP}
    # Return value of `fatrop_ocp_c_solve`, `nothing` if not solved
    solve_status::Union{Nothing,Cint}
    stats::Union{Nothing,FatropOcpCStats}
    # Exception thrown by a callback
    exception::Any
    # Results in the order of Fatrop's primal vector and equality constraints
    primal::Vector{Float64}
    dual::Vector{Float64}
    objective_value::Float64
    solve_time::Float64
    function Optimizer()
        return new(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            Dict{String,Any}(),
            false,
            nothing,
            nothing,
            nothing,
            nothing,
            Float64[],
            Float64[],
            NaN,
            NaN,
        )
    end
end

function _reset_results(model::Optimizer)
    model.ocp = nothing
    model.solve_status = nothing
    model.stats = nothing
    model.exception = nothing
    empty!(model.primal)
    empty!(model.dual)
    model.objective_value = NaN
    model.solve_time = NaN
    return
end

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.model)
    _reset_results(model)
    return
end

MOI.is_empty(model::Optimizer) = MOI.is_empty(model.model)

MOI.get(::Optimizer, ::MOI.SolverName) = "Fatrop"

MOI.supports_incremental_interface(::Optimizer) = false

function _check_attributes(dest, src, index_type, attrs)
    for attr in attrs
        if !MOI.supports(dest, attr, index_type...)
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    return
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    # `dest.model` supports any attribute so we need to check the attributes
    # supported by `dest` here
    _check_attributes(
        dest,
        src,
        (),
        MOI.get(src, MOI.ListOfModelAttributesSet()),
    )
    _check_attributes(
        dest,
        src,
        (MOI.VariableIndex,),
        MOI.get(src, MOI.ListOfVariableAttributesSet()),
    )
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !MOI.supports_constraint(dest, F, S)
            throw(MOI.UnsupportedConstraint{F,S}())
        end
        _check_attributes(
            dest,
            src,
            (MOI.ConstraintIndex{F,S},),
            MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}()),
        )
    end
    return MOI.copy_to(dest.model, src)
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{<:_SCALAR_FUNCTIONS})
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:_SCALAR_FUNCTIONS},
    ::Type{<:_SCALAR_SETS},
)
    return true
end

MOI.supports(::Optimizer, ::MOI.Name) = true

MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{MOI.VariableIndex}) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintName,
    ::Type{<:MOI.ConstraintIndex},
)
    return true
end

MOI.supports(::Optimizer, ::Stage, ::Type{MOI.VariableIndex}) = true

MOI.supports(::Optimizer, ::MOI.UserDefinedFunction) = true

function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, value)
    MOI.set(model.model, attr, value)
    return
end

function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
    nlp = MOI.Nonlinear.Model()
    _register_operators(nlp, model.model)
    return MOI.get(nlp, attr)
end

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

const _MODEL_ATTRIBUTES = Union{
    MOI.Name,
    MOI.ObjectiveSense,
    MOI.ObjectiveFunctionType,
    MOI.ObjectiveFunction,
    MOI.NumberOfVariables,
    MOI.ListOfVariableIndices,
    MOI.ListOfConstraintTypesPresent,
    MOI.NumberOfConstraints,
    MOI.ListOfConstraintIndices,
    MOI.ListOfModelAttributesSet,
    MOI.ListOfVariableAttributesSet,
    MOI.ListOfConstraintAttributesSet,
}

MOI.get(model::Optimizer, attr::_MODEL_ATTRIBUTES) = MOI.get(model.model, attr)

const _INDEX_ATTRIBUTES = Union{
    MOI.VariableName,
    MOI.VariablePrimalStart,
    Stage,
    MOI.ConstraintName,
    MOI.ConstraintFunction,
    MOI.ConstraintSet,
}

function MOI.get(model::Optimizer, attr::_INDEX_ATTRIBUTES, index::MOI.Index)
    return MOI.get(model.model, attr, index)
end

function MOI.set(
    model::Optimizer,
    attr::Union{Stage,MOI.VariablePrimalStart},
    vi::MOI.VariableIndex,
    value,
)
    MOI.set(model.model, attr, vi, value)
    return
end

function MOI.is_valid(model::Optimizer, index::MOI.Index)
    return MOI.is_valid(model.model, index)
end

### Options

MOI.supports(::Optimizer, ::MOI.Silent) = true

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    if !haskey(model.options, attr.name)
        throw(MOI.GetAttributeNotAllowed(attr))
    end
    return model.options[attr.name]
end

const _OPTION_TYPES = (Float64, Int, Bool, String)

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    # Fatrop throws a C++ exception through the C interface for an unknown
    # option or an option of the wrong type, which would abort Julia.
    type = fatrop_ocp_c_option_type(attr.name)
    if type == -1
        throw(MOI.UnsupportedAttribute(attr, "Unknown option of Fatrop."))
    end
    model.options[attr.name] = convert(_OPTION_TYPES[type+1], value)
    return
end

### Callbacks

function _optimizer(user_data::Ptr{Cvoid})
    return unsafe_pointer_to_objref(user_data)::Optimizer
end
_ocp(user_data::Ptr{Cvoid}) = _optimizer(user_data).ocp::_OCP

_get_nx(k::Cint, user_data::Ptr{Cvoid}) = Cint(_ocp(user_data).nx[k+1])
_get_nu(k::Cint, user_data::Ptr{Cvoid}) = Cint(_ocp(user_data).nu[k+1])
_get_ng(k::Cint, user_data::Ptr{Cvoid}) = Cint(_ocp(user_data).ng[k+1])
function _get_ng_ineq(k::Cint, user_data::Ptr{Cvoid})
    return Cint(_ocp(user_data).ng_ineq[k+1])
end
_get_horizon_length(user_data::Ptr{Cvoid}) = Cint(_ocp(user_data).K)

# Julia exceptions must not unwind through the C++ frames of Fatrop so we catch
# them, store the first one and rethrow it at the end of `MOI.optimize!`.
function _catch(f::Function, user_data::Ptr{Cvoid}, ret::Cint)
    try
        f(_ocp(user_data))
    catch err
        model = _optimizer(user_data)
        if isnothing(model.exception)
            model.exception = CapturedException(err, catch_backtrace())
        end
        return Cint(1)
    end
    return ret
end

function _get_bounds(
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    k::Cint,
    user_data::Ptr{Cvoid},
)
    return _catch(user_data, Cint(0)) do ocp
        for i in 1:ocp.ng_ineq[k+1]
            row = ocp.rows[ocp.ineq_offsets[k+1]+i]
            unsafe_store!(lower, row.lower, i)
            unsafe_store!(upper, row.upper, i)
        end
    end
end

function _get_initial(
    ptr::Ptr{Cdouble},
    k::Cint,
    user_data::Ptr{Cvoid},
    offset::Function,
    n::Function,
)
    return _catch(user_data, Cint(0)) do ocp
        model = _optimizer(user_data)
        for i in 1:n(ocp, k+1)
            vi = ocp.variables[ocp.offsets[k+1]+offset(ocp, k+1)+i]
            start = MOI.get(model.model, MOI.VariablePrimalStart(), vi)
            unsafe_store!(ptr, something(start, 0.0), i)
        end
    end
end

function _get_initial_uk(uk::Ptr{Cdouble}, k::Cint, user_data::Ptr{Cvoid})
    return _get_initial(uk, k, user_data, (_, _) -> 0, (ocp, k) -> ocp.nu[k])
end

function _get_initial_xk(xk::Ptr{Cdouble}, k::Cint, user_data::Ptr{Cvoid})
    return _get_initial(
        xk,
        k,
        user_data,
        (ocp, k) -> ocp.nu[k],
        (ocp, k) -> ocp.nx[k],
    )
end

function _primal(ocp::_OCP, x::Ptr{Cdouble})
    return unsafe_wrap(Array, x, length(ocp.variables))
end

# Return value of the `full_eval_*` callbacks telling Fatrop that the stagewise
# callbacks should not be called. This is what CasADi returns. Note that `2`
# would also skip the stagewise callbacks but `eval_constraint_violation` would
# then also skip subtracting the slack variables from the residual of the
# inequality constraints.
const _FULLY_EVALUATED = Cint(1)

function _full_eval_lag_hess(
    objective_scale::Cdouble,
    primal_x::Ptr{Cdouble},
    lam::Ptr{Cdouble},
    ::Ptr{Cdouble},
    ::Ptr{Cdouble},
    RSQrqt::Ptr{blasfeo_dmat},
    ::Ptr{Cvoid},
    user_data::Ptr{Cvoid},
)
    return _catch(user_data, _FULLY_EVALUATED) do ocp
        λ = unsafe_wrap(Array, lam, length(ocp.rows))
        _eval_hessian_lagrangian(
            ocp,
            _primal(ocp, primal_x),
            objective_scale,
            λ,
        )
        for k in 1:ocp.K
            blasfeo_pack_dmat(ocp.RSQ[k], _stage_matrix(RSQrqt, k - 1))
        end
    end
end

function _full_eval_constr_jac(
    primal_x::Ptr{Cdouble},
    ::Ptr{Cdouble},
    ::Ptr{Cdouble},
    BAbt::Ptr{blasfeo_dmat},
    Gg_eqt::Ptr{blasfeo_dmat},
    Gg_ineqt::Ptr{blasfeo_dmat},
    ::Ptr{Cvoid},
    user_data::Ptr{Cvoid},
)
    return _catch(user_data, _FULLY_EVALUATED) do ocp
        _eval_constraint_jacobian(ocp, _primal(ocp, primal_x))
        for k in 1:ocp.K
            if k < ocp.K
                blasfeo_pack_dmat(ocp.BA[k], _stage_matrix(BAbt, k - 1))
            end
            blasfeo_pack_dmat(ocp.Gg[k], _stage_matrix(Gg_eqt, k - 1))
            blasfeo_pack_dmat(ocp.Gg_ineq[k], _stage_matrix(Gg_ineqt, k - 1))
        end
    end
end

function _full_eval_contr_viol(
    primal_x::Ptr{Cdouble},
    ::Ptr{Cdouble},
    ::Ptr{Cdouble},
    res::Ptr{Cdouble},
    ::Ptr{Cvoid},
    user_data::Ptr{Cvoid},
)
    return _catch(user_data, _FULLY_EVALUATED) do ocp
        r = unsafe_wrap(Array, res, length(ocp.rows))
        return _eval_constraint_violation(ocp, r, _primal(ocp, primal_x))
    end
end

function _full_eval_obj_grad(
    objective_scale::Cdouble,
    primal_x::Ptr{Cdouble},
    ::Ptr{Cdouble},
    ::Ptr{Cdouble},
    res::Ptr{Cdouble},
    ::Ptr{Cvoid},
    user_data::Ptr{Cvoid},
)
    return _catch(user_data, _FULLY_EVALUATED) do ocp
        grad = unsafe_wrap(Array, res, length(ocp.variables))
        _eval_objective_gradient(ocp, grad, _primal(ocp, primal_x))
        return grad .*= objective_scale
    end
end

function _full_eval_obj(
    objective_scale::Cdouble,
    primal_x::Ptr{Cdouble},
    ::Ptr{Cdouble},
    ::Ptr{Cdouble},
    res::Ptr{Cdouble},
    ::Ptr{Cvoid},
    user_data::Ptr{Cvoid},
)
    return _catch(user_data, _FULLY_EVALUATED) do ocp
        value = _eval_objective(ocp, _primal(ocp, primal_x))
        return unsafe_store!(res, objective_scale * value)
    end
end

function _write(msg::Ptr{Cchar}, num::Cint)
    print(stdout, unsafe_string(msg, num))
    return
end

_flush() = flush(stdout)

function _interface(model::Optimizer)
    return FatropOcpCInterface(
        @cfunction(_get_nx, Cint, (Cint, Ptr{Cvoid})),
        @cfunction(_get_nu, Cint, (Cint, Ptr{Cvoid})),
        @cfunction(_get_ng, Cint, (Cint, Ptr{Cvoid})),
        C_NULL, # get_n_stage_params
        C_NULL, # get_n_global_params
        C_NULL, # get_default_stage_params
        C_NULL, # get_default_global_params
        @cfunction(_get_ng_ineq, Cint, (Cint, Ptr{Cvoid})),
        @cfunction(_get_horizon_length, Cint, (Ptr{Cvoid},)),
        # The stagewise evaluations are never called as the `full_eval_*`
        # callbacks return `_FULLY_EVALUATED`.
        C_NULL, # eval_BAbt
        C_NULL, # eval_RSQrqt
        C_NULL, # eval_Ggt
        C_NULL, # eval_Ggt_ineq
        C_NULL, # eval_b
        C_NULL, # eval_g
        C_NULL, # eval_gineq
        C_NULL, # eval_rq
        C_NULL, # eval_L
        @cfunction(
            _get_bounds,
            Cint,
            (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid})
        ),
        @cfunction(_get_initial_xk, Cint, (Ptr{Cdouble}, Cint, Ptr{Cvoid})),
        @cfunction(_get_initial_uk, Cint, (Ptr{Cdouble}, Cint, Ptr{Cvoid})),
        @cfunction(
            _full_eval_lag_hess,
            Cint,
            (
                Cdouble,
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{blasfeo_dmat},
                Ptr{Cvoid},
                Ptr{Cvoid},
            )
        ),
        @cfunction(
            _full_eval_constr_jac,
            Cint,
            (
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{blasfeo_dmat},
                Ptr{blasfeo_dmat},
                Ptr{blasfeo_dmat},
                Ptr{Cvoid},
                Ptr{Cvoid},
            )
        ),
        @cfunction(
            _full_eval_contr_viol,
            Cint,
            (
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cvoid},
                Ptr{Cvoid},
            )
        ),
        @cfunction(
            _full_eval_obj_grad,
            Cint,
            (
                Cdouble,
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cvoid},
                Ptr{Cvoid},
            )
        ),
        @cfunction(
            _full_eval_obj,
            Cint,
            (
                Cdouble,
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cdouble},
                Ptr{Cvoid},
                Ptr{Cvoid},
            )
        ),
        pointer_from_objref(model),
    )
end

function MOI.optimize!(model::Optimizer)
    start_time = time()
    _reset_results(model)
    ocp = _OCP(model.model)
    model.ocp = ocp
    interface = Ref(_interface(model))
    GC.@preserve model interface begin
        solver = fatrop_ocp_c_create(
            interface,
            @cfunction(_write, Cvoid, (Ptr{Cchar}, Cint)),
            @cfunction(_flush, Cvoid, ())
        )
        try
            if model.silent
                fatrop_ocp_c_set_option(solver, "print_level", 0)
            end
            for (name, value) in model.options
                fatrop_ocp_c_set_option(solver, name, value)
            end
            model.solve_status = fatrop_ocp_c_solve(solver)
            model.stats = fatrop_ocp_c_get_stats(solver)
            model.primal = zeros(length(ocp.variables))
            blasfeo_unpack_dvec!(model.primal, fatrop_ocp_c_get_primal(solver))
            model.dual = zeros(length(ocp.rows))
            blasfeo_unpack_dvec!(model.dual, fatrop_ocp_c_get_dual(solver))
        finally
            fatrop_ocp_c_destroy(solver)
        end
    end
    if !isnothing(model.exception)
        throw(model.exception)
    end
    if ocp.has_objective
        model.objective_value = MOI.eval_objective(ocp.evaluator, model.primal)
    else
        model.objective_value = 0.0
    end
    model.solve_time = time() - start_time
    return
end

### Results

function _flag(model::Optimizer)
    if isnothing(model.solve_status) || model.solve_status == -1
        return
    end
    return IpSolverReturnFlag(model.stats.return_flag)
end

const _TERMINATION_STATUS = Dict(
    Success => MOI.LOCALLY_SOLVED,
    MaxIterExceeded => MOI.ITERATION_LIMIT,
    StopAtAcceptablePoint => MOI.ALMOST_LOCALLY_SOLVED,
    LocalInfeasibility => MOI.LOCALLY_INFEASIBLE,
    LineSearchFailed => MOI.SLOW_PROGRESS,
    ErrorInStepComputation => MOI.NUMERICAL_ERROR,
    Unknown => MOI.OTHER_ERROR,
)

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if isnothing(model.solve_status)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    flag = _flag(model)
    if isnothing(flag)
        return MOI.OTHER_ERROR
    end
    return _TERMINATION_STATUS[flag]
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if isnothing(model.solve_status)
        return "Optimize not called"
    end
    flag = _flag(model)
    if isnothing(flag)
        return "Exception thrown by Fatrop"
    end
    return string(flag)
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return isnothing(_flag(model)) ? 0 : 1
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1 || isnothing(_flag(model))
        return MOI.NO_SOLUTION
    end
    flag = _flag(model)
    if flag == Success
        return MOI.FEASIBLE_POINT
    elseif flag == StopAtAcceptablePoint
        return MOI.NEARLY_FEASIBLE_POINT
    elseif flag == LocalInfeasibility
        return MOI.INFEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if attr.result_index != 1 || isnothing(_flag(model))
        return MOI.NO_SOLUTION
    end
    flag = _flag(model)
    if flag == Success
        return MOI.FEASIBLE_POINT
    elseif flag == StopAtAcceptablePoint
        return MOI.NEARLY_FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

function MOI.get(model::Optimizer, ::MOI.BarrierIterations)
    return Int(model.stats.iterations_count)
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.objective_value
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return model.primal[model.ocp.position[vi]]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    f = MOI.get(model.model, MOI.ConstraintFunction(), ci)
    return MOI.Utilities.eval_variables(model.model, f) do vi
        return MOI.get(model, MOI.VariablePrimal(attr.result_index), vi)
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    r = get(model.ocp.row_of, ci, nothing)
    if isnothing(r)
        # The constraint is trivially satisfied so it was not given to Fatrop
        return 0.0
    end
    return -model.ocp.rows[r].scale * model.dual[r]
end
