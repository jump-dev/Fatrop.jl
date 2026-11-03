# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# Low-level wrapper of the C interface of Fatrop declared in
# `c_interface/include/fatrop/ocp/OCPCInterface.h`.

const fatrop_int = Cint

# `struct blasfeo_dmat` for the panel-major format `MF_PANELMAJ` that
# `blasfeo_jll` is built with. We never access its fields, we only need its
# size because Fatrop gives us a pointer to the first element of a C array of
# such structs, one per stage.
struct blasfeo_dmat
    mem::Ptr{Cdouble}
    pA::Ptr{Cdouble}
    dA::Ptr{Cdouble}
    m::Cint
    n::Cint
    pm::Cint
    cn::Cint
    use_dA::Cint
    memsize::Cint
end

function _stage_matrix(mats::Ptr{blasfeo_dmat}, k::Integer)
    return mats + k * sizeof(blasfeo_dmat)
end

# Copy the column-major `A` into the top-left corner of the matrix `sB`
function blasfeo_pack_dmat(A::Matrix{Cdouble}, sB::Ptr{blasfeo_dmat})
    m, n = size(A)
    ccall(
        (:blasfeo_pack_dmat, blasfeo_jll.blasfeo),
        Cvoid,
        (Cint, Cint, Ptr{Cdouble}, Cint, Ptr{blasfeo_dmat}, Cint, Cint),
        m,
        n,
        A,
        max(1, m),
        sB,
        0,
        0,
    )
    return
end

function blasfeo_unpack_dvec!(y::Vector{Cdouble}, sx::Ptr{Cvoid})
    ccall(
        (:blasfeo_unpack_dvec, blasfeo_jll.blasfeo),
        Cvoid,
        (Cint, Ptr{Cvoid}, Cint, Ptr{Cdouble}, Cint),
        length(y),
        sx,
        0,
        y,
        1,
    )
    return y
end

struct FatropOcpCInterface
    get_nx::Ptr{Cvoid}
    get_nu::Ptr{Cvoid}
    get_ng::Ptr{Cvoid}
    get_n_stage_params::Ptr{Cvoid}
    get_n_global_params::Ptr{Cvoid}
    get_default_stage_params::Ptr{Cvoid}
    get_default_global_params::Ptr{Cvoid}
    get_ng_ineq::Ptr{Cvoid}
    get_horizon_length::Ptr{Cvoid}
    eval_BAbt::Ptr{Cvoid}
    eval_RSQrqt::Ptr{Cvoid}
    eval_Ggt::Ptr{Cvoid}
    eval_Ggt_ineq::Ptr{Cvoid}
    eval_b::Ptr{Cvoid}
    eval_g::Ptr{Cvoid}
    eval_gineq::Ptr{Cvoid}
    eval_rq::Ptr{Cvoid}
    eval_L::Ptr{Cvoid}
    get_bounds::Ptr{Cvoid}
    get_initial_xk::Ptr{Cvoid}
    get_initial_uk::Ptr{Cvoid}
    full_eval_lag_hess::Ptr{Cvoid}
    full_eval_constr_jac::Ptr{Cvoid}
    full_eval_contr_viol::Ptr{Cvoid}
    full_eval_obj_grad::Ptr{Cvoid}
    full_eval_obj::Ptr{Cvoid}
    user_data::Ptr{Cvoid}
end

struct FatropOcpCStats
    compute_sd_time::Cdouble
    duinf_time::Cdouble
    eval_hess_time::Cdouble
    eval_jac_time::Cdouble
    eval_cv_time::Cdouble
    eval_grad_time::Cdouble
    eval_obj_time::Cdouble
    initialization_time::Cdouble
    time_total::Cdouble
    eval_hess_count::Cint
    eval_jac_count::Cint
    eval_cv_count::Cint
    eval_grad_count::Cint
    eval_obj_count::Cint
    iterations_count::Cint
    return_flag::Cint
end

# Values of `fatrop::IpSolverReturnFlag`, the `return_flag` of the stats
@enum(
    IpSolverReturnFlag,
    Success,
    MaxIterExceeded,
    StopAtAcceptablePoint,
    LocalInfeasibility,
    LineSearchFailed,
    ErrorInStepComputation,
    Unknown,
)

function fatrop_ocp_c_create(
    ocp::Ref{FatropOcpCInterface},
    write::Ptr{Cvoid},
    flush::Ptr{Cvoid},
)
    return ccall(
        (:fatrop_ocp_c_create, libfatrop),
        Ptr{Cvoid},
        (Ref{FatropOcpCInterface}, Ptr{Cvoid}, Ptr{Cvoid}),
        ocp,
        write,
        flush,
    )
end

function fatrop_ocp_c_solve(solver::Ptr{Cvoid})
    return ccall(
        (:fatrop_ocp_c_solve, libfatrop),
        fatrop_int,
        (Ptr{Cvoid},),
        solver,
    )
end

function fatrop_ocp_c_destroy(solver::Ptr{Cvoid})
    ccall((:fatrop_ocp_c_destroy, libfatrop), Cvoid, (Ptr{Cvoid},), solver)
    return
end

"""
    fatrop_ocp_c_option_type(name::String)

Return `-1` if the option is unknown, `0` for `Float64`, `1` for `Int`, `2`
for `Bool` and `3` for `String`.
"""
function fatrop_ocp_c_option_type(name::String)
    return ccall((:fatrop_ocp_c_option_type, libfatrop), Cint, (Cstring,), name)
end

function fatrop_ocp_c_set_option(solver::Ptr{Cvoid}, name::String, value::Real)
    return ccall(
        (:fatrop_ocp_c_set_option_double, libfatrop),
        Cint,
        (Ptr{Cvoid}, Cstring, Cdouble),
        solver,
        name,
        value,
    )
end

function fatrop_ocp_c_set_option(
    solver::Ptr{Cvoid},
    name::String,
    value::Integer,
)
    return ccall(
        (:fatrop_ocp_c_set_option_int, libfatrop),
        Cint,
        (Ptr{Cvoid}, Cstring, Cint),
        solver,
        name,
        value,
    )
end

function fatrop_ocp_c_set_option(solver::Ptr{Cvoid}, name::String, value::Bool)
    return ccall(
        (:fatrop_ocp_c_set_option_bool, libfatrop),
        Cint,
        (Ptr{Cvoid}, Cstring, Cint),
        solver,
        name,
        value,
    )
end

function fatrop_ocp_c_set_option(
    solver::Ptr{Cvoid},
    name::String,
    value::String,
)
    return ccall(
        (:fatrop_ocp_c_set_option_string, libfatrop),
        Cint,
        (Ptr{Cvoid}, Cstring, Cstring),
        solver,
        name,
        value,
    )
end

function fatrop_ocp_c_get_primal(solver::Ptr{Cvoid})
    return ccall(
        (:fatrop_ocp_c_get_primal, libfatrop),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        solver,
    )
end

function fatrop_ocp_c_get_dual(solver::Ptr{Cvoid})
    return ccall(
        (:fatrop_ocp_c_get_dual, libfatrop),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        solver,
    )
end

function fatrop_ocp_c_get_stats(solver::Ptr{Cvoid})
    ptr = ccall(
        (:fatrop_ocp_c_get_stats, libfatrop),
        Ptr{FatropOcpCStats},
        (Ptr{Cvoid},),
        solver,
    )
    return unsafe_load(ptr)
end
