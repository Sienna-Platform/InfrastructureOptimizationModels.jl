# NMDT bilinear approximations of z = x·y.
#   NMDTBilinearConfig  — discretize x only; y is just normalized.
#   DNMDTBilinearConfig — discretize both x and y, combine two estimates.

"""
Config for single-NMDT bilinear approximation (discretizes x only).

# Fields
- `depth::Int`: number of binary discretization levels L for x.
"""
struct NMDTBilinearConfig <: BilinearApproxConfig
    depth::Int
end

"""
Config for double-NMDT bilinear approximation (discretizes both x and y).

# Fields
- `depth::Int`: number of binary discretization levels L for both x and y.
"""
struct DNMDTBilinearConfig <: BilinearApproxConfig
    depth::Int
end

# --- Scalar build (pure JuMP) ---

"""
    build_bilinear_approx(config::NMDTBilinearConfig, model, x, y, x_min, x_max, y_min, y_max)

Scalar form: approximate x·y via NMDT for one cell. Discretize x, normalize
y to yh ∈ [0,1], build the binary-continuous product β·yh and residual
δ·yh, reassemble x·y.

Returns `(; approximation, x_discretization, yh_expression, bx_yh_product,
residual_product)`.
"""
function build_bilinear_approx(
    config::NMDTBilinearConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    x_disc = build_discretization(model, x, x_min, x_max, config.depth)
    yh_expr = (y - y_min) / (y_max - y_min)
    bx_yh = build_binary_continuous_product(
        model, x_disc.beta_var, yh_expr, 0.0, 1.0, config.depth,
    )
    dz = build_residual_product(
        model, x_disc.delta_var, yh_expr, 1.0, config.depth,
    )
    approximation = build_assembled_product(
        model, [bx_yh.result_expression], dz.z_var,
        x_disc.norm_expr, yh_expr,
        x_min, x_max, y_min, y_max,
    )
    return (;
        approximation,
        x_discretization = x_disc,
        yh_expression = yh_expr,
        bx_yh_product = bx_yh,
        residual_product = dz,
    )
end

"""
    build_bilinear_approx(config::DNMDTBilinearConfig, model, x, y, x_min, x_max, y_min, y_max)

Scalar form: DNMDT bilinear approximation at one cell. Discretize both x
and y, form all four cross binary-continuous products, convex-combine.
"""
function build_bilinear_approx(
    config::DNMDTBilinearConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    x_disc = build_discretization(model, x, x_min, x_max, config.depth)
    y_disc = build_discretization(model, y, y_min, y_max, config.depth)
    bx_yh = build_binary_continuous_product(
        model, x_disc.beta_var, y_disc.norm_expr, 0.0, 1.0, config.depth,
    )
    by_dx = build_binary_continuous_product(
        model, y_disc.beta_var, x_disc.delta_var,
        0.0, 2.0^(-config.depth), config.depth,
    )
    by_xh = build_binary_continuous_product(
        model, y_disc.beta_var, x_disc.norm_expr, 0.0, 1.0, config.depth,
    )
    bx_dy = build_binary_continuous_product(
        model, x_disc.beta_var, y_disc.delta_var,
        0.0, 2.0^(-config.depth), config.depth,
    )
    dz = build_residual_product(
        model, x_disc.delta_var, y_disc.delta_var,
        2.0^(-config.depth), config.depth,
    )
    approximation = build_assembled_dnmdt(
        model,
        bx_yh.result_expression,
        by_dx.result_expression,
        by_xh.result_expression,
        bx_dy.result_expression,
        dz.z_var,
        x_disc.norm_expr, y_disc.norm_expr,
        x_min, x_max, y_min, y_max,
    )
    return (;
        approximation,
        x_discretization = x_disc,
        y_discretization = y_disc,
        bx_yh_product = bx_yh,
        by_dx_product = by_dx,
        by_xh_product = by_xh,
        bx_dy_product = bx_dy,
        residual_product = dz,
    )
end

# --- IOM adapter ---

"""
    add_bilinear_approx!(config::NMDTBilinearConfig, container, ::Type{C}, x_var, y_var, x_bounds, y_bounds, meta)

Allocate x discretization + yh expression + binary-continuous-product +
residual-product + BilinearProductExpression containers. Loop `(name, t)`.
"""
function add_bilinear_approx!(
    config::NMDTBilinearConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    depth = config.depth
    @assert length(name_axis) == length(x_bounds)
    @assert length(name_axis) == length(y_bounds)
    for i in eachindex(x_bounds)
        @assert x_bounds[i].max > x_bounds[i].min
        @assert y_bounds[i].max > y_bounds[i].min
    end

    model = get_jump_model(container)

    x_disc_targets = _alloc_discretization_targets!(
        container, C, name_axis, time_axis, depth, meta * "_x",
    )
    yh_target = add_expression_container!(
        container, NormedVariableExpression, C, name_axis, time_axis;
        meta = meta * "_y",
    )
    bx_yh_targets = _alloc_binary_continuous_product_targets!(
        container, C, name_axis, time_axis, depth, meta; tighten = false,
    )
    res_targets = _alloc_residual_product_targets!(
        container, C, name_axis, time_axis, meta; tighten = false,
    )
    approx_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        ymn, ymx = y_bounds[i].min, y_bounds[i].max
        for t in time_axis
            r = build_bilinear_approx(
                config, model, x_var[name, t], y_var[name, t], xmn, xmx, ymn, ymx,
            )
            _write_discretization_cell!(x_disc_targets, name, t, r.x_discretization, depth)
            yh_target[name, t] = r.yh_expression
            _write_binary_continuous_product_cell!(
                bx_yh_targets,
                name,
                t,
                r.bx_yh_product,
                depth,
            )
            _write_residual_product_cell!(res_targets, name, t, r.residual_product)
            approx_target[name, t] = r.approximation
        end
    end
    return approx_target
end

"""
    add_bilinear_approx!(config::DNMDTBilinearConfig, container, ::Type{C}, x_var, y_var, x_bounds, y_bounds, meta)

Allocate two discretizations + four binary-continuous-product + one
residual-product + BilinearProductExpression containers. Loop `(name, t)`.
"""
function add_bilinear_approx!(
    config::DNMDTBilinearConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    depth = config.depth
    @assert length(name_axis) == length(x_bounds)
    @assert length(name_axis) == length(y_bounds)
    for i in eachindex(x_bounds)
        @assert x_bounds[i].max > x_bounds[i].min
        @assert y_bounds[i].max > y_bounds[i].min
    end

    model = get_jump_model(container)

    x_disc_targets = _alloc_discretization_targets!(
        container, C, name_axis, time_axis, depth, meta * "_x",
    )
    y_disc_targets = _alloc_discretization_targets!(
        container, C, name_axis, time_axis, depth, meta * "_y",
    )
    bx_yh_targets = _alloc_binary_continuous_product_targets!(
        container, C, name_axis, time_axis, depth, meta * "_bx_yh"; tighten = false,
    )
    by_dx_targets = _alloc_binary_continuous_product_targets!(
        container, C, name_axis, time_axis, depth, meta * "_by_dx"; tighten = false,
    )
    by_xh_targets = _alloc_binary_continuous_product_targets!(
        container, C, name_axis, time_axis, depth, meta * "_by_xh"; tighten = false,
    )
    bx_dy_targets = _alloc_binary_continuous_product_targets!(
        container, C, name_axis, time_axis, depth, meta * "_bx_dy"; tighten = false,
    )
    res_targets = _alloc_residual_product_targets!(
        container, C, name_axis, time_axis, meta; tighten = false,
    )
    approx_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        ymn, ymx = y_bounds[i].min, y_bounds[i].max
        for t in time_axis
            r = build_bilinear_approx(
                config, model, x_var[name, t], y_var[name, t], xmn, xmx, ymn, ymx,
            )
            _write_discretization_cell!(x_disc_targets, name, t, r.x_discretization, depth)
            _write_discretization_cell!(y_disc_targets, name, t, r.y_discretization, depth)
            _write_binary_continuous_product_cell!(
                bx_yh_targets,
                name,
                t,
                r.bx_yh_product,
                depth,
            )
            _write_binary_continuous_product_cell!(
                by_dx_targets,
                name,
                t,
                r.by_dx_product,
                depth,
            )
            _write_binary_continuous_product_cell!(
                by_xh_targets,
                name,
                t,
                r.by_xh_product,
                depth,
            )
            _write_binary_continuous_product_cell!(
                bx_dy_targets,
                name,
                t,
                r.bx_dy_product,
                depth,
            )
            _write_residual_product_cell!(res_targets, name, t, r.residual_product)
            approx_target[name, t] = r.approximation
        end
    end
    return approx_target
end
