# SOS2-based piecewise linear approximation of x² for use in constraints.
# A single formulation parameterized by the adjacency-enforcement backend:
#   - SolverBackend: solver-native MOI.SOS2 constraints
#   - ManualBackend: binary segment-selection variables + linear adjacency constraints
# The lambda/link/norm/result construction is shared; only the adjacency step differs.

"lambda_var (λ) convex combination weight variables for SOS2 quadratic approximation."
struct QuadraticVariable <: SparseVariableType end
"Links x to the weighted sum of breakpoints in SOS2 quadratic approximation."
struct SOS2LinkingConstraint <: ConstraintType end
"Expression for the weighted sum of breakpoints Σ λ_i * x_i linking x to lambda variables."
struct SOS2LinkingExpression <: ExpressionType end
"Ensures the sum of λ weights equals 1 in SOS2 quadratic approximation."
struct SOS2NormConstraint <: ConstraintType end
"Expression for the normalization sum Σ λ_i in SOS2 quadratic approximation."
struct SOS2NormExpression <: ExpressionType end

"Solver-native MOI.SOS2 adjacency constraint on lambda variables."
struct SolverSOS2Constraint <: ConstraintType end

"Binary segment-selection variables (z) for manual SOS2 quadratic approximation."
struct ManualSOS2BinaryVariable <: SparseVariableType end
"Ensures exactly one segment is active (∑zⱼ = 1) in manual SOS2 quadratic approximation."
struct ManualSOS2SegmentSelectionConstraint <: ConstraintType end
"Expression for the segment selection sum Σ z_j in manual SOS2 quadratic approximation."
struct ManualSOS2SegmentSelectionExpression <: ExpressionType end
"Links active segment to lambda variables."
struct ManualSOS2AdjacencyConstraint <: ConstraintType end

"""
Config for SOS2 piecewise linear quadratic approximation.

The `B <: SOS2Backend` type parameter selects how SOS2 adjacency is enforced:
- `SolverBackend`: solver-native `MOI.SOS2` constraints.
- `ManualBackend`: binary segment-selection variables with linear adjacency constraints.

# Fields
- `depth::Int`: number of PWL segments (breakpoints = depth + 1)
- `tightener::Tightener`: optional relaxation strengthener (default `NoTightener()`). The
  supported tightener is `McCormickTightener` (piecewise-McCormick cuts on the concave `−v²`
  term); `partitions` must evenly divide `depth`.

The worst-case PWL overestimation gap is `Δ²/(4·depth²)`. A `McCormickTightener` adds
piecewise-McCormick cuts on the concave relaxation surface; it does **not** change the
MIP-optimal worst-case error, only the LP relaxation given to branch-and-bound.

**Constraint:** `McCormickTightener.partitions` must evenly divide `depth` (enforced by the
constructor). The PWMCC chord upper bound is valid for the PWL value `q` only when every PWMCC
boundary coincides with a PWL breakpoint, which on a uniform grid holds iff partitions divides
`depth`; otherwise a chord straddling a breakpoint cuts off MIP-feasible solutions (e.g.
`depth=3, partitions=2` makes `v ∈ (4/9, 5/9)` infeasible). See
`tolerance_depth(::Type{<:SOS2QuadConfig}; …)` to derive `depth` from a target tolerance.
"""
struct SOS2QuadConfig{B <: SOS2Backend} <: QuadraticApproxConfig
    depth::Int
    tightener::Tightener

    function SOS2QuadConfig{B}(;
        depth::Int,
        tightener::Tightener = NoTightener(),
    ) where {B <: SOS2Backend}
        supports_tightener(SOS2QuadConfig{B}, tightener) || throw(
            ArgumentError("SOS2QuadConfig does not support tightener $(typeof(tightener))"),
        )
        _assert_partitions_divide_depth(tightener, depth)
        return new{B}(depth, tightener)
    end
end

"SOS2 piecewise linear approximation over-estimates x² at the segment midpoints."
sidedness(::Type{<:SOS2QuadConfig}) = OneSidedOver()

"SOS2 supports piecewise-McCormick (`McCormickTightener`) on its concave term."
supports_tightener(::Type{<:SOS2QuadConfig}, ::McCormickTightener) = true

"""
    tolerance_depth(::Type{<:SOS2QuadConfig}; tolerance, max_delta)::Int

Smallest SOS2 segment count `d` whose worst-case PWL gap on `[a, a+Δ]` falls
within `tolerance`. Inverts `Δ²/(4·d²) ≤ τ`:
```
d = ⌈Δ / (2·√τ)⌉
```
clamped to `d ≥ 1`. `pwmcc_segments` does not enter the error bound, so it is
left to the constructor. Backend-independent.
"""
function tolerance_depth(
    ::Type{<:SOS2QuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    _check_tolerance_args(tolerance, max_delta)
    return _ceil_positive(max_delta / (2 * sqrt(tolerance)))
end

# --- Backend-specific adjacency enforcement ---
#
# `_sos2_adjacency_containers!` creates the containers a backend needs (once, up
# front) and returns them as a NamedTuple. `_sos2_adjacency!` adds the per-(name, t)
# adjacency constraints using the lambda vector built in the shared loop.

function _sos2_adjacency_containers!(
    ::SolverBackend,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    n_points::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    return (
        sos_cons = add_constraints_container!(
            container,
            SolverSOS2Constraint,
            C,
            names,
            time_steps;
            meta,
        ),
    )
end

function _sos2_adjacency!(
    ::SolverBackend,
    jump_model::JuMP.Model,
    conts,
    ::Type{C},
    name::String,
    t::Int,
    lambda::Vector{JuMP.VariableRef},
    n_points::Int,
) where {C <: IS.InfrastructureSystemsComponent}
    conts.sos_cons[name, t] =
        JuMP.@constraint(jump_model, lambda in MOI.SOS2(collect(1:n_points)))
    return
end

function _sos2_adjacency_containers!(
    ::ManualBackend,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    n_points::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    n_bins = n_points - 1
    return (
        z_container = add_variable_container!(
            container,
            ManualSOS2BinaryVariable,
            C,
            names,
            1:n_bins,
            time_steps;
            meta,
        ),
        seg_cons = add_constraints_container!(
            container,
            ManualSOS2SegmentSelectionConstraint,
            C,
            names,
            time_steps;
            meta,
        ),
        seg_expr = add_expression_container!(
            container,
            ManualSOS2SegmentSelectionExpression,
            C,
            names,
            time_steps;
            meta,
        ),
        adj_cons = add_constraints_container!(
            container,
            ManualSOS2AdjacencyConstraint,
            C,
            names,
            1:n_points,
            time_steps;
            meta,
        ),
    )
end

function _sos2_adjacency!(
    ::ManualBackend,
    jump_model::JuMP.Model,
    conts,
    ::Type{C},
    name::String,
    t::Int,
    lambda::Vector{JuMP.VariableRef},
    n_points::Int,
) where {C <: IS.InfrastructureSystemsComponent}
    n_bins = n_points - 1

    # Create binary segment-selection variables z_j
    z_vars = Vector{JuMP.VariableRef}(undef, n_bins)
    for j in 1:n_bins
        z_vars[j] =
            conts.z_container[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "ManualSOS2Binary_$(C)_{$(name), $(j), $(t)}",
                binary = true,
            )
    end

    # Σ z_j = 1 (segment selection)
    seg = conts.seg_expr[name, t] = JuMP.@expression(jump_model, sum(z_vars))
    conts.seg_cons[name, t] = JuMP.@constraint(jump_model, seg == 1)

    # Adjacency constraints: λ_i ≤ z_{i-1} + z_i (with boundary cases)
    # λ_1 ≤ z_1
    conts.adj_cons[name, 1, t] = JuMP.@constraint(jump_model, lambda[1] <= z_vars[1])
    # λ_i ≤ z_{i-1} + z_i for i = 2..n-1
    for i in 2:(n_points - 1)
        conts.adj_cons[name, i, t] =
            JuMP.@constraint(jump_model, lambda[i] <= z_vars[i - 1] + z_vars[i])
    end
    # λ_n ≤ z_{n-1}
    conts.adj_cons[name, n_points, t] =
        JuMP.@constraint(jump_model, lambda[n_points] <= z_vars[n_bins])
    return
end

"""
    add_quadratic_approx!(config::SOS2QuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Approximate x² using a piecewise linear function with SOS2 adjacency.

Creates lambda_var (λ) variables representing convex combination weights over breakpoints,
adds linking, normalization, and backend-specific adjacency constraints, and stores affine
expressions approximating x² in a `QuadraticExpression` expression container. The backend
(`SolverBackend` → `MOI.SOS2`; `ManualBackend` → binary segment selection) is selected by the
config's type parameter.

# Arguments
- `config::SOS2QuadConfig`: configuration with `depth` (number of PWL segments) and `tightener` (`McCormickTightener` for PWMCC cuts, or `NoTightener()`)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)
"""
function add_quadratic_approx!(
    config::SOS2QuadConfig{B},
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {B <: SOS2Backend, C <: IS.InfrastructureSystemsComponent}
    x_bkpts, x_sq_bkpts =
        _get_breakpoints_for_pwl_function(
            0.0,
            1.0,
            _square;
            num_segments = config.depth,
        )
    n_points = config.depth + 1
    jump_model = get_jump_model(container)

    # Create all shared containers upfront
    lambda_var = add_variable_container!(
        container,
        QuadraticVariable,
        C,
        names,
        1:n_points,
        time_steps;
        meta,
    )
    link_cons = add_constraints_container!(
        container,
        SOS2LinkingConstraint,
        C,
        names,
        time_steps;
        meta,
    )
    link_expr = add_expression_container!(
        container,
        SOS2LinkingExpression,
        C,
        names,
        time_steps;
        meta,
    )
    norm_cons = add_constraints_container!(
        container,
        SOS2NormConstraint,
        C,
        names,
        time_steps;
        meta,
    )
    norm_expr = add_expression_container!(
        container,
        SOS2NormExpression,
        C,
        names,
        time_steps;
        meta,
    )
    adj = _sos2_adjacency_containers!(B(), container, C, names, time_steps, n_points, meta)
    result_expr = add_expression_container!(
        container,
        QuadraticExpression,
        C,
        names,
        time_steps;
        meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        b = bounds[i]
        IS.@assert_op b.max > b.min
        lx = b.max - b.min
        x = x_var[name, t]

        # Create lambda variables: λ_i ∈ [0, 1]
        lambda = Vector{JuMP.VariableRef}(undef, n_points)
        for k in 1:n_points
            lambda[k] =
                lambda_var[name, k, t] = JuMP.@variable(
                    jump_model,
                    base_name = "QuadraticVariable_$(C)_{$(name), pwl_$(k), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = 1.0,
                )
        end

        # x = Σ λ_i * x_i
        link =
            link_expr[name, t] =
                JuMP.@expression(
                    jump_model,
                    sum(x_bkpts[k] * lambda[k] for k in eachindex(x_bkpts))
                )
        link_cons[name, t] = JuMP.@constraint(jump_model, (x - b.min) / lx == link)

        # Σ λ_i = 1
        norm = norm_expr[name, t] = JuMP.@expression(jump_model, sum(lambda))
        norm_cons[name, t] = JuMP.@constraint(jump_model, norm == 1.0)

        # Backend-specific adjacency enforcement
        _sos2_adjacency!(B(), jump_model, adj, C, name, t, lambda, n_points)

        # Build x̂² = Σ λ_i * x_i², then unnormalize: x² = lx²·x̂² + 2·x_min·x − x_min²
        x_hat_sq =
            JuMP.@expression(jump_model, sum(x_sq_bkpts[k] * lambda[k] for k in 1:n_points))
        result_expr[name, t] =
            JuMP.@expression(jump_model, lx * lx * x_hat_sq + 2 * b.min * x - b.min * b.min)
    end

    apply_tightener!(
        config.tightener, config, container, C, names, time_steps,
        x_var, result_expr, bounds, meta,
    )

    return result_expr
end

"Apply PWMCC concave cuts on the SOS2 `−v²` term (valid inequality, preserves the MIP optimum)."
function apply_tightener!(
    t::McCormickTightener,
    ::SOS2QuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    result_expr,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    _add_pwmcc_concave_cuts!(
        container, C, names, time_steps,
        x_var, result_expr, bounds,
        t.partitions, meta * "_pwmcc";
        backend = t.backend,
    )
    return
end
