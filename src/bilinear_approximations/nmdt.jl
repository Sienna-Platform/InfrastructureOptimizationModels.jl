# DNMDT (Double Normalized Multiparametric Disaggregation Technique) bilinear approximation of x·y.
# Independently discretizes both x and y, forms four cross binary-continuous products, then
# combines two NMDT estimates with a convex weighting λ (default 0.5). Reduces to the NMDT
# formulation when applied to x·x (quadratic case).
# Reference: Teles, Castro, Matos (2013), Multiparametric disaggregation technique for global
# optimization of polynomial programming problems.

"""
Config for NMDT bilinear approximation, parameterized by the variant `V <: NMDTVariant`:
`SingleNMDT` discretizes only x; `DoubleNMDT` (DNMDT) discretizes both x and y.

# Fields
- `depth::Int`: number of binary discretization levels L

The worst-case relaxation gap on `[ax, ax+Δx] × [ay, ay+Δy]` is `Δx·Δy·2^{-2L-2}` for
`DoubleNMDT` and `Δx·Δy·2^{-L-2}` for `SingleNMDT`. See
`tolerance_depth(::Type{NMDTBilinearConfig{V}}; …)` to derive `depth` from a target tolerance.
"""
struct NMDTBilinearConfig{V <: NMDTVariant} <: BilinearApproxConfig
    depth::Int

    NMDTBilinearConfig{V}(; depth::Int) where {V <: NMDTVariant} = new{V}(depth)
end

"""
    tolerance_depth(::Type{NMDTBilinearConfig{DoubleNMDT}}; tolerance, max_delta_x, max_delta_y)::Int

Smallest DNMDT bilinear depth `L` whose worst-case relaxation gap on
`[ax, ax+Δx] × [ay, ay+Δy]` falls within `tolerance`. Inverts `Δx·Δy·2^{-2L-2} ≤ τ`:
`L = ⌈(log₂(Δx·Δy/τ) − 2) / 2⌉`, clamped to `L ≥ 1`.
"""
function tolerance_depth(
    ::Type{NMDTBilinearConfig{DoubleNMDT}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
)
    _check_tolerance_args(tolerance, max_delta_x, max_delta_y)
    return _ceil_positive((log2(max_delta_x * max_delta_y / tolerance) - 2) / 2)
end

"""
    tolerance_depth(::Type{NMDTBilinearConfig{SingleNMDT}}; tolerance, max_delta_x, max_delta_y)::Int

Smallest single-NMDT bilinear depth `L` whose worst-case relaxation gap on
`[ax, ax+Δx] × [ay, ay+Δy]` falls within `tolerance`. Inverts `Δx·Δy·2^{-L-2} ≤ τ`:
`L = ⌈log₂(Δx·Δy/τ) − 2⌉`, clamped to `L ≥ 1`.
"""
function tolerance_depth(
    ::Type{NMDTBilinearConfig{SingleNMDT}};
    tolerance::Float64,
    max_delta_x::Float64,
    max_delta_y::Float64,
)
    _check_tolerance_args(tolerance, max_delta_x, max_delta_y)
    return _ceil_positive(log2(max_delta_x * max_delta_y / tolerance) - 2)
end

# --- DNMDT bilinear approximation ---

"""
    _bilinear_from_discretization!(config::NMDTBilinearConfig{DoubleNMDT}, container, C, names, time_steps, x_disc, y_disc, x_bounds, y_bounds, meta)

Approximate x·y using the DNMDT method from pre-built discretizations.

Constructs all four cross binary-continuous products (β_x·yh, β_y·δx, β_y·xh, β_x·δy)
then delegates to the core DNMDT assembler. Stores results in a `BilinearProductExpression`
container.

# Arguments
- `config::NMDTBilinearConfig{DoubleNMDT}`: configuration for the DNMDT bilinear approximation
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `y_disc::NMDTDiscretization`: pre-built discretization for y
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _bilinear_from_discretization!(
    config::NMDTBilinearConfig{DoubleNMDT},
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    bx_yh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, y_disc.norm_expr, 0.0, 1.0,
        config.depth, meta * "_bx_yh",
    )
    by_dx_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        y_disc, x_disc.delta_var, 0.0, 2.0^(-config.depth),
        config.depth, meta * "_by_dx",
    )
    by_xh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        y_disc, x_disc.norm_expr, 0.0, 1.0,
        config.depth, meta * "_by_xh",
    )
    bx_dy_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, y_disc.delta_var, 0.0, 2.0^(-config.depth),
        config.depth, meta * "_bx_dy",
    )

    return _assemble_dnmdt!(
        container, C, names, time_steps,
        bx_yh_expr, by_dx_expr, by_xh_expr, bx_dy_expr,
        x_disc, y_disc, x_bounds, y_bounds,
        config.depth, meta; result_type = BilinearProductExpression,
    )
end

"""
    add_bilinear_approx!(config::NMDTBilinearConfig{DoubleNMDT}, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Approximate x·y using the DNMDT method from raw variable inputs.

Discretizes both x and y independently via `_discretize!` then delegates to the
pre-discretized overload.

# Arguments
- `config::NMDTBilinearConfig{DoubleNMDT}`: configuration for the DNMDT bilinear approximation
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_bilinear_approx!(
    config::NMDTBilinearConfig{DoubleNMDT},
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
    x_disc = _discretize!(
        container,
        C,
        names,
        time_steps,
        x_var,
        x_bounds,
        config.depth,
        meta * "_x",
    )
    y_disc = _discretize!(
        container,
        C,
        names,
        time_steps,
        y_var,
        y_bounds,
        config.depth,
        meta * "_y",
    )
    return _bilinear_from_discretization!(
        config,
        container,
        C,
        names,
        time_steps,
        x_disc,
        y_disc,
        x_bounds,
        y_bounds,
        meta,
    )
end

# --- NMDT bilinear approximation ---

"""
    _bilinear_from_discretization!(config::NMDTBilinearConfig{SingleNMDT}, container, C, names, time_steps, x_disc, yh_expr, x_bounds, y_bounds, meta)

Approximate x·y using the NMDT method from a pre-built x discretization and normalized y.

Discretizes only x (using `x_disc`) while y is already normalized to yh ∈ [0,1].
Computes binary-continuous product β_x·yh and residual product δ_x·yh, then assembles
x·y via `_assemble_product!`. Stores results in a `BilinearProductExpression` container.

# Arguments
- `config::NMDTBilinearConfig{SingleNMDT}`: configuration for the NMDT bilinear approximation
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `yh_expr`: expression container for the normalized variable yh = (y − y_min)/(y_max − y_min)
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _bilinear_from_discretization!(
    config::NMDTBilinearConfig{SingleNMDT},
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    yh_expr,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    bx_y_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, yh_expr, 0.0, 1.0,
        config.depth, meta,
    )
    dz = _residual_product!(
        container, C, names, time_steps,
        x_disc, yh_expr, 1.0, config.depth, meta;
    )

    return _assemble_product!(
        container, C, names, time_steps,
        [bx_y_expr], dz,
        x_disc, yh_expr, x_bounds, y_bounds,
        meta; result_type = BilinearProductExpression,
    )
end

"""
    add_bilinear_approx!(config::NMDTBilinearConfig{SingleNMDT}, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Approximate x·y using the NMDT method from raw variable inputs.

Discretizes x via `_discretize!` and normalizes y via `_normed_variable!`, then
delegates to the pre-discretized overload.

# Arguments
- `config::NMDTBilinearConfig{SingleNMDT}`: configuration for the NMDT bilinear approximation
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_bilinear_approx!(
    config::NMDTBilinearConfig{SingleNMDT},
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
    x_disc = _discretize!(
        container,
        C,
        names,
        time_steps,
        x_var,
        x_bounds,
        config.depth,
        meta * "_x",
    )
    yh_expr =
        _normed_variable!(container, C, names, time_steps, y_var, y_bounds, meta * "_y")
    return _bilinear_from_discretization!(
        config,
        container,
        C,
        names,
        time_steps,
        x_disc,
        yh_expr,
        x_bounds,
        y_bounds,
        meta,
    )
end
