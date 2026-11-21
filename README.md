# Fatrop.jl

Fatrop.jl is a wrapper for [Fatrop](https://github.com/meco-group/fatrop), a
nonlinear optimal control problem solver exploiting the structure of the
problem with a Riccati-based linear solver.

## License

`Fatrop.jl` is licensed under the [MIT license](LICENSE.md).

The underlying solver, [meco-group/fatrop](https://github.com/meco-group/fatrop),
is licensed under the [BSD 2-Clause license](https://github.com/meco-group/fatrop/blob/main/LICENSE.txt),
except for some files derived from Ipopt which are licensed under the
[EPL 2.0 license](https://github.com/meco-group/fatrop/blob/main/LICENSE-EPL-2.0.txt).

## Installation

Fatrop is not available in Yggdrasil yet. Build the shared library `libfatrop`
with its C interface (`-DBUILD_WITH_C_INTERFACE=ON`) against the BLASFEO of
`blasfeo_jll` and set the `JULIA_FATROP_LIBRARY_PATH` environment variable to
its path before loading Fatrop.jl.

## Use with JuMP

Fatrop solves problems of the form
```
min  Σₖ lₖ(uₖ, xₖ)
s.t. xₖ₊₁ = fₖ(uₖ, xₖ)
     hₖ(uₖ, xₖ) = 0
     Lₖ ≤ gₖ(uₖ, xₖ) ≤ Uₖ
```
Communicate the stage of each variable with the `Fatrop.Stage` attribute. The
states and dynamics are then detected automatically: an equality constraint
involving variables of two consecutive stages must be of the form
`x[k+1][i] == f(u[k], x[k])` where `x[k+1][i]` is a single variable of stage
`k + 1` appearing affinely. All other constraints and the Hessian of the
objective must only involve variables of a single stage.

```julia
using JuMP, Fatrop
N, dt = 20, 0.1
model = Model(Fatrop.Optimizer)
@variable(model, p[1:N+1])
@variable(model, v[1:N+1])
@variable(model, -1 <= u[1:N] <= 1)
for k in 1:N+1
    set_attribute(p[k], Fatrop.Stage(), k)
    set_attribute(v[k], Fatrop.Stage(), k)
    if k <= N
        set_attribute(u[k], Fatrop.Stage(), k)
    end
end
@constraint(model, p[1] == 0)
@constraint(model, v[1] == 0)
@constraint(model, [k in 1:N], p[k+1] == p[k] + dt * v[k])
@constraint(model, [k in 1:N], v[k+1] == v[k] + dt * sin(u[k]))
@constraint(model, p[N+1] == 0.5)
@constraint(model, v[N+1] == 0)
@objective(model, Min, sum(u[k]^2 for k in 1:N))
optimize!(model)
```

If the `Fatrop.Stage` attribute is not set, all variables are in the same stage
so any nonlinear program can be solved but no structure is exploited.

## Options

Options are set with `set_attribute(model, name, value)`, for instance
`set_attribute(model, "max_iter", 100)` or `set_attribute(model, "tol", 1e-6)`.
