# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestMOIWrapper

using Test
import Fatrop
import MathOptInterface as MOI

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

function test_runtests()
    model = MOI.instantiate(
        Fatrop.Optimizer;
        with_bridge_type = Float64,
        with_cache_type = Float64,
    )
    MOI.set(model, MOI.Silent(), true)
    config = MOI.Test.Config(;
        atol = 1e-4,
        rtol = 1e-4,
        optimal_status = MOI.LOCALLY_SOLVED,
        infeasible_status = MOI.LOCALLY_INFEASIBLE,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ObjectiveBound,
            MOI.DualObjectiveValue,
            # The C interface of Fatrop does not give its version
            MOI.SolverVersion,
        ],
    )
    MOI.Test.runtests(
        model,
        config;
        exclude = Regex[
            # Fatrop returns `LineSearchFailed` instead of detecting the local
            # infeasibility
            r"^test_conic_linear_INFEASIBLE$",
            r"^test_conic_linear_INFEASIBLE_2$",
            r"^test_conic_NormInfinityCone_INFEASIBLE$",
            r"^test_conic_NormOneCone_INFEASIBLE$",
            r"^test_linear_INFEASIBLE$",
            r"^test_linear_INFEASIBLE_2$",
            r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_",
            # Fatrop does not detect unboundedness
            r"^test_linear_DUAL_INFEASIBLE$",
            r"^test_linear_DUAL_INFEASIBLE_2$",
            r"^test_solve_TerminationStatus_DUAL_INFEASIBLE$",
            # When the equality constraints fully determine the variables and
            # there are inequality constraints, Fatrop reaches the iteration
            # limit with a zero step and a constant dual infeasibility
            r"^test_nonlinear_with_scalar_quadratic_function_with_off_diag$",
            r"^test_quadratic_nonconvex_constraint_basic$",
            # Fatrop requires the Hessian but the user-defined operator only
            # provides its gradient
            r"^test_nonlinear_expression_multivariate_function$",
            # The constraint `x - x == 0` has a zero Jacobian
            r"^test_linear_complex_Zeros_duplicate$",
        ],
    )
    return
end

end  # module TestMOIWrapper

TestMOIWrapper.runtests()
