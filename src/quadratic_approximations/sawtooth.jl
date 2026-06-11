# Sawtooth MIP approximation of x² for use in constraints.
# Uses recursive tooth function compositions with O(log(1/ε)) binary variables.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024).

"Auxiliary continuous variables (g₀, …, g_L) for sawtooth quadratic approximation."
struct SawtoothAuxVariable <: VariableType end
"Binary variables (α₁, …, α_L) for sawtooth quadratic approximation."
struct SawtoothBinaryVariable <: VariableType end
"Variable result in tightened version."
struct SawtoothTightenedVariable <: VariableType end
"Links g₀ to the normalized x value in sawtooth quadratic approximation."
struct SawtoothLinkingConstraint <: ConstraintType end
"Constrains g_j based on g_{j-1}."
struct SawtoothMIPConstraint <: ConstraintType end
"LP relaxation constraints (g_j ≤ 2g_{j-1}, g_j ≤ 2(1−g_{j-1})) used in epigraph tightening."
struct SawtoothLPConstraint <: ConstraintType end
"Bounds tightened variable."
struct SawtoothTightenedConstraint <: ConstraintType end

"""
Config for sawtooth MIP quadratic approximation.

# Fields
- `depth::Int`: recursion depth L; uses L binary variables for 2^L + 1 breakpoints
- `epigraph_depth::Int`: depth of an additional epigraph Q^{L1} lower bound (0 disables, default 0)

## Effect of `epigraph_depth` on the worst-case error

`epigraph_depth = 0`: the formulation is purely deterministic — `result_expr =
sawtooth(x)` is pinned by the binary structure, so `|z − x²| ≤ Δ²·2^{-2L-2}`
where Δ = max − min.

`epigraph_depth = L_e > 0`: a free continuous variable `z` is introduced with
`z ≤ sawtooth(x)` and `z ≥ epigraph(x)`, and `result_expr = z`. This is a
*structural* change, not just LP-tightening: the MIP-feasible set of
`result_expr` values grows from a single point to the interval
`[epigraph(x), sawtooth(x)]`. The worst-case error over that interval is
```
|z − x²| ≤ max(Δ²·2^{-2L-2}, Δ²·2^{-2L_e-4})
```
- `L_e ≥ L − 1`: sawtooth dominates, worst-case = `Δ²·2^{-2L-2}`.
- `L_e < L − 1`: epigraph dominates, worst-case = `Δ²·2^{-2L_e-4}`.

Contrast with `pwmcc_segments` on the SOS2 variants, which adds genuine LP cuts
and never changes the MIP-feasible set.

See `tolerance_depth(::Type{SawtoothQuadConfig}; …)` to derive `depth` from a
target tolerance, and `tolerance_epigraph_depth(::Type{SawtoothQuadConfig}; …)`
for the matching `epigraph_depth`.
"""
struct SawtoothQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int

    SawtoothQuadConfig(; depth::Int, epigraph_depth::Int = 0) =
        new(depth, epigraph_depth)
end

"""
    tolerance_depth(::Type{SawtoothQuadConfig}; tolerance, max_delta)::Int

Smallest sawtooth depth `L` whose worst-case overestimation gap on `[a, a+Δ]`
falls within `tolerance`. Inverts the closed-form bound `Δ²·2^{-2L-2} ≤ τ`:
```
L = ⌈(log₂(Δ²/τ) − 2) / 2⌉
```
clamped to `L ≥ 1`. Sizes only the sawtooth side of the formulation.

**Contract on `epigraph_depth`**: the returned depth meets the tolerance iff
the user picks `epigraph_depth = 0` (tightening disabled) or
`epigraph_depth ≥ depth − 1`. When `0 < epigraph_depth < depth − 1`, the
epigraph side of the sandwich has a larger error than the sawtooth side
(epigraph `Δ²·2^{-2L_e-4}` vs sawtooth `Δ²·2^{-2L-2}`), and since
`result_expr` is free in `[epigraph(x), sawtooth(x)]` in larger optimization
contexts the realized error can exceed `tolerance`. Use
`tolerance_epigraph_depth` below to size both knobs consistently.
"""
function tolerance_depth(
    ::Type{SawtoothQuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    _check_tolerance_args(tolerance, max_delta)
    return _ceil_positive((log2(max_delta^2 / tolerance) - 2) / 2)
end

"""
    tolerance_epigraph_depth(::Type{SawtoothQuadConfig}; tolerance, max_delta)::Int

Smallest `epigraph_depth` consistent with `tolerance_depth(SawtoothQuadConfig; …)`
under target tolerance `τ`. Returns the depth at which the epigraph (LP lower)
side of the sandwich has worst-case error ≤ τ on `[a, a+Δ]`. Since the
epigraph's per-unit error is `Δ²·2^{-2L_e-4}` versus the sawtooth's
`Δ²·2^{-2L-2}`, this is one less than the sawtooth depth for the same
tolerance — which exactly meets the contract `epigraph_depth ≥ depth − 1`.

Optional: callers that want to disable epigraph tightening can pass
`epigraph_depth = 0` and the sawtooth-side bound still holds.
"""
function tolerance_epigraph_depth(
    ::Type{SawtoothQuadConfig};
    tolerance::Float64,
    max_delta::Float64,
)
    return tolerance_depth(EpigraphQuadConfig; tolerance, max_delta)
end

"""
    _add_quadratic_approx!(config::SawtoothQuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Approximate x² using the sawtooth MIP formulation.

Creates auxiliary continuous variables g_0,...,g_L and binary variables α_1,...,α_L,
adds S^L constraints (4 per level) and a linking constraint for each component and
time step, and stores affine expressions approximating x² in a
`QuadraticExpression` expression container.

For depth L, the approximation interpolates x² at 2^L + 1 uniformly spaced breakpoints
with maximum overestimation error Δ² · 2^{-2L-2} where Δ = x_max - x_min.

# Arguments
- `config::SawtoothQuadConfig`: configuration with `depth` (recursion depth L; uses L binary variables) and `epigraph_depth` (LP tightening depth; 0 to disable)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name lower and upper bounds of x domain
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)
"""
function _add_quadratic_approx!(
    config::SawtoothQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op config.depth >= 1
    jump_model = get_jump_model(container)

    # Create containers with known dimensions
    g_levels = 0:(config.depth)
    alpha_levels = 1:(config.depth)
    g_var = add_variable_container!(
        container,
        SawtoothAuxVariable,
        C,
        names,
        g_levels,
        time_steps;
        meta,
    )
    alpha_var = add_variable_container!(
        container,
        SawtoothBinaryVariable,
        C,
        names,
        alpha_levels,
        time_steps;
        meta,
    )
    mip_cons = add_constraints_container!(
        container,
        SawtoothMIPConstraint,
        C,
        names,
        alpha_levels,
        1:4,
        time_steps;
        sparse = true,
        meta,
    )
    link_cons = add_constraints_container!(
        container,
        SawtoothLinkingConstraint,
        C,
        names,
        time_steps;
        meta,
    )
    result_expr = add_expression_container!(
        container,
        QuadraticExpression,
        C,
        names,
        time_steps;
        meta,
    )

    if config.epigraph_depth > 0
        lp_expr = _add_quadratic_approx!(
            EpigraphQuadConfig(; depth = config.epigraph_depth),
            container, C, names, time_steps,
            x_var, bounds, meta * "_lb",
        )
        z_var = add_variable_container!(
            container,
            SawtoothTightenedVariable,
            C,
            names,
            time_steps;
            meta,
        )
        tight_cons = add_constraints_container!(
            container,
            SawtoothTightenedConstraint,
            C,
            names,
            1:2,
            time_steps;
            meta,
        )
    end

    for (i, name) in enumerate(names), t in time_steps
        b = bounds[i]
        IS.@assert_op b.max > b.min
        delta = b.max - b.min
        saw_coeffs = [delta * delta * (2.0^(-2 * j)) for j in alpha_levels]
        z_min = (b.min <= 0.0 <= b.max) ? 0.0 : min(b.min * b.min, b.max * b.max)
        z_max = max(b.min * b.min, b.max * b.max)
        x = x_var[name, t]

        # Auxiliary variables g_0,...,g_L ∈ [0, 1]
        for j in g_levels
            g_var[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "SawtoothAux_$(C)_{$(name), $(j), $(t)}",
                lower_bound = 0.0,
                upper_bound = 1.0,
            )
        end

        # Binary variables α_1,...,α_L
        for j in alpha_levels
            alpha_var[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "SawtoothBin_$(C)_{$(name), $(j), $(t)}",
                binary = true,
            )
        end

        # Linking constraint: g_0 = (x - x_min) / Δ
        link_cons[name, t] = JuMP.@constraint(
            jump_model,
            g_var[name, 0, t] == (x - b.min) / delta,
        )

        # S^L constraints for j = 1,...,L
        for j in alpha_levels
            g_prev = g_var[name, j - 1, t]
            g_curr = g_var[name, j, t]
            alpha_j = alpha_var[name, j, t]

            # g_j ≤ 2 g_{j-1}
            mip_cons[name, j, 1, t] = JuMP.@constraint(jump_model, g_curr <= 2.0 * g_prev)
            # g_j ≤ 2(1 - g_{j-1})
            mip_cons[name, j, 2, t] =
                JuMP.@constraint(jump_model, g_curr <= 2.0 * (1.0 - g_prev))
            # g_j ≥ 2(g_{j-1} - α_j)
            mip_cons[name, j, 3, t] =
                JuMP.@constraint(jump_model, g_curr >= 2.0 * (g_prev - alpha_j))
            # g_j ≥ 2(α_j - g_{j-1})
            mip_cons[name, j, 4, t] =
                JuMP.@constraint(jump_model, g_curr >= 2.0 * (alpha_j - g_prev))
        end

        # Build x² ≈ x_min² + (2 x_min Δ + Δ²) g_0 - Σ_{j=1}^L Δ² 2^{-2j} g_j
        x_sq_approx = JuMP.AffExpr(b.min * b.min)
        add_proportional_to_jump_expression!(
            x_sq_approx,
            g_var[name, 0, t],
            2.0 * b.min * delta + delta * delta,
        )
        for j in alpha_levels
            add_proportional_to_jump_expression!(
                x_sq_approx,
                g_var[name, j, t],
                -saw_coeffs[j],
            )
        end

        if config.epigraph_depth > 0
            z =
                z_var[name, t] = JuMP.@variable(
                    jump_model,
                    base_name = "TightenedSawtooth_$(C)_{$(name), $(t)}",
                    lower_bound = z_min,
                    upper_bound = z_max
                )
            tight_cons[name, 1, t] = JuMP.@constraint(jump_model, z <= x_sq_approx)
            tight_cons[name, 2, t] = JuMP.@constraint(jump_model, z >= lp_expr[name, t])
            result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
        else
            result_expr[name, t] = x_sq_approx
        end
    end

    return result_expr
end
