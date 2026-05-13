# Shared infrastructure for quadratic and bilinear approximations.
#
# This file defines the two abstract config supertypes, the two abstract
# result supertypes, the generic IOM wrappers that POM calls into, and a
# handful of expression key types used by multiple methods.
#
# The architecture is layered:
#
#   Pure-JuMP layer  (the math)
#     build_quadratic_approx(config, model, x, bounds) -> QuadraticApproxResult
#     build_bilinear_approx(config, model, x, y, x_bounds, y_bounds) -> BilinearApproxResult
#
#   IOM layer  (container bookkeeping)
#     _add_quadratic_approx!(config, container, C, names, time_steps, x, bounds, meta)
#       1. call build_quadratic_approx
#       2. dispatch register_in_container!(container, C, result, meta) to write
#          all auxiliary JuMP objects into the OptimizationContainer
#       3. return the approximation expression container
#
# Each method file under src/approximations/ contains its own config struct,
# result struct, build_* function, and register_in_container! method.

# --- Abstract config and result supertypes ---

"Abstract supertype for quadratic approximation method configurations."
abstract type QuadraticApproxConfig end

"Abstract supertype for bilinear approximation method configurations."
abstract type BilinearApproxConfig end

"Abstract supertype for the pure-JuMP result of a quadratic approximation build."
abstract type QuadraticApproxResult end

"Abstract supertype for the pure-JuMP result of a bilinear approximation build."
abstract type BilinearApproxResult end

"""
    get_approximation(result)

Return the approximation expression container from a quadratic or bilinear
approximation result. The container is indexed by (name, time_step) and
holds either `JuMP.AffExpr` or `JuMP.QuadExpr` entries depending on method.
"""
get_approximation(result::QuadraticApproxResult) = result.approximation
get_approximation(result::BilinearApproxResult) = result.approximation

# --- Shared expression-key types ---

"Expression container for the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1]."
struct NormedVariableExpression <: ExpressionType end

"Expression container for quadratic (x²) approximation results."
struct QuadraticExpression <: ExpressionType end

"Expression container for bilinear product (x·y) approximation results."
struct BilinearProductExpression <: ExpressionType end

"Variable container for bilinear product (x·y) approximation results."
struct BilinearProductVariable <: VariableType end

"Expression container for sums of two variables, x + y."
struct VariableSumExpression <: ExpressionType end

"Expression container for differences of two variables, x − y."
struct VariableDifferenceExpression <: ExpressionType end

# --- Pure-JuMP helper: normalized variable ---

"""
    build_normed_variable(model, x, bounds) -> DenseAxisArray{JuMP.AffExpr, 2}

Build a 2D container of affine expressions xh = (x − x_min) / (x_max − x_min) ∈ [0,1].

Pure-JuMP utility used by methods that operate on a normalized variable
(NMDT, and any caller that needs a [0,1] domain).

# Arguments
- `model`: JuMP model the expressions live in (only needed for axis types).
- `x`: 2D JuMP container indexed by (name, t).
- `bounds`: per-name bounds aligned with the first axis of `x`.
"""
function build_normed_variable(
    model::JuMP.Model,
    x,
    bounds::Vector{MinMax},
)
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(bounds)
    for b in bounds
        IS.@assert_op b.max > b.min
    end
    slope = JuMP.Containers.DenseAxisArray(
        [1.0 / (b.max - b.min) for b in bounds],
        name_axis,
    )
    offset = JuMP.Containers.DenseAxisArray(
        [-b.min / (b.max - b.min) for b in bounds],
        name_axis,
    )
    return JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        slope[name] * x[name, t] + offset[name],
    )
end

# --- IOM-side wrappers (POM entry points) ---

"""
    _add_quadratic_approx!(config, container, C, names, time_steps, x_var, bounds, meta)

POM entry point for quadratic approximation. Dispatched on the abstract
`QuadraticApproxConfig` type — concrete behavior comes from the concrete
config's `build_quadratic_approx` and `register_in_container!` methods.

# Arguments
- `config::QuadraticApproxConfig`: approximation method configuration.
- `container::OptimizationContainer`: the optimization container.
- `::Type{C}`: component type (used for container key dispatch).
- `names::Vector{String}`: component names; must equal `axes(x_var, 1)`.
- `time_steps::UnitRange{Int}`: time periods; must equal `axes(x_var, 2)`.
- `x_var`: 2D JuMP container indexed by (name, t).
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of the x domain.
- `meta::String`: identifier used to disambiguate container keys when more
  than one approximation of the same kind is registered on the same component
  type. The IOM wrapper passes this through to `register_in_container!`.

# Returns
The approximation expression container (indexed by (name, t)), as returned
by `get_approximation(result)`.
"""
function _add_quadratic_approx!(
    config::QuadraticApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result = build_quadratic_approx(config, get_jump_model(container), x_var, bounds)
    register_in_container!(container, C, result, meta)
    return get_approximation(result)
end

"""
    _add_bilinear_approx!(config, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

POM entry point for bilinear approximation. Dispatched on the abstract
`BilinearApproxConfig` type — concrete behavior comes from the concrete
config's `build_bilinear_approx` and `register_in_container!` methods.

# Arguments
- `config::BilinearApproxConfig`: approximation method configuration.
- `container::OptimizationContainer`: the optimization container.
- `::Type{C}`: component type (used for container key dispatch).
- `names::Vector{String}`: component names; must equal `axes(x_var, 1)` and `axes(y_var, 1)`.
- `time_steps::UnitRange{Int}`: time periods.
- `x_var`, `y_var`: 2D JuMP containers indexed by (name, t).
- `x_bounds`, `y_bounds`: per-name lower and upper bounds.
- `meta::String`: identifier used to disambiguate container keys.

# Returns
The approximation expression container (indexed by (name, t)).
"""
function _add_bilinear_approx!(
    config::BilinearApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result = build_bilinear_approx(
        config,
        get_jump_model(container),
        x_var,
        y_var,
        x_bounds,
        y_bounds,
    )
    register_in_container!(container, C, result, meta)
    return get_approximation(result)
end

# --- register_in_container! interface ---
# Concrete methods are defined in each method file, dispatched on result-struct type.

"""
    register_in_container!(container, ::Type{C}, result, meta)

Write the JuMP objects held by `result` into the `OptimizationContainer`
using the appropriate key types and `meta` suffix.

Each concrete result struct provides its own method. The math layer
(`build_*`) never references container keys directly — that name → key
mapping lives only inside `register_in_container!`.
"""
function register_in_container! end
