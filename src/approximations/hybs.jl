# HybS (Hybrid Separable) MIP relaxation for bilinear products z = x·y.
# Combines a Bin2 lower bound and a Bin3 upper bound with shared quadratic
# approximations of x², y² and pure-LP epigraph relaxations of (x+y)², (x−y)².
# Uses 2L binaries instead of Bin2's 3L.
# Reference: Beach, Burlacu, Bärmann, Hager, Hildebrand (2024), Definition 10.

"Two-sided HybS bound constraints: Bin2 lower + Bin3 upper."
struct HybSBoundConstraint <: ConstraintType end

"""
Config for HybS bilinear approximation.

# Fields
- `quad_config::QuadraticApproxConfig`: quadratic method used for x² and y².
- `epigraph_depth::Int`: depth of the epigraph Q^{L1} approximation of (x±y)².
- `add_mccormick::Bool`: whether to add a standard McCormick envelope on z (default false).
"""
struct HybSConfig{QC <: QuadraticApproxConfig} <: BilinearApproxConfig
    quad_config::QC
    epigraph_depth::Int
    add_mccormick::Bool
end
function HybSConfig(quad_config::QuadraticApproxConfig, epigraph_depth::Int)
    return HybSConfig(quad_config, epigraph_depth, false)
end

"""
    build_bilinear_approx(config::HybSConfig, model, x, y, x_min, x_max, y_min, y_max)

Scalar form: build x² and y² via the chosen quadratic method, build
(x+y)² and (x−y)² via the epigraph Q^{L1} relaxation, and constrain a
fresh product variable z with two-sided bounds derived from the Bin2 lower
/ Bin3 upper identities.

Returns `(; approximation, xsq, ysq, sum_expression, diff_expression,
sum_epigraph, diff_epigraph, z_var, bound_constraints, mccormick_constraints)`.
"""
function build_bilinear_approx(
    config::HybSConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    xsq = build_quadratic_approx(config.quad_config, model, x, x_min, x_max)
    ysq = build_quadratic_approx(config.quad_config, model, y, y_min, y_max)
    return _build_hybs_scalar(
        config, model, x, y, xsq, ysq, x_min, x_max, y_min, y_max,
    )
end

# Shared math between the standard and precomputed-form scalar entrypoints.
# `xsq`/`ysq` are NamedTuples (or any object with an `.approximation` field).
function _build_hybs_scalar(
    config::HybSConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    xsq,
    ysq,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    p1_expr = JuMP.@expression(model, x + y)
    p2_expr = JuMP.@expression(model, x - y)
    p1_min, p1_max = x_min + y_min, x_max + y_max
    p2_min, p2_max = x_min - y_max, x_max - y_min

    epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
    zp1 = build_quadratic_approx(epi_cfg, model, p1_expr, p1_min, p1_max)
    zp2 = build_quadratic_approx(epi_cfg, model, p2_expr, p2_min, p2_max)

    z_lo = min(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)
    z_hi = max(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)
    z_var = JuMP.@variable(
        model, lower_bound = z_lo, upper_bound = z_hi, base_name = "HybSProduct",
    )

    bound_1 = JuMP.@constraint(
        model,
        z_var >= 0.5 * (zp1.approximation - xsq.approximation - ysq.approximation),
    )
    bound_2 = JuMP.@constraint(
        model,
        z_var <= 0.5 * (xsq.approximation + ysq.approximation - zp2.approximation),
    )
    bound_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(undef, 1:2)
    bound_cons[1] = bound_1
    bound_cons[2] = bound_2

    approximation = JuMP.@expression(model, 1.0 * z_var)

    mc = if config.add_mccormick
        build_mccormick_envelope(
        model, x, y, z_var, x_min, x_max, y_min, y_max,
    )
    else
        nothing
    end

    return (;
        approximation,
        xsq,
        ysq,
        sum_expression = p1_expr,
        diff_expression = p2_expr,
        sum_epigraph = zp1,
        diff_epigraph = zp2,
        z_var,
        bound_constraints = bound_cons,
        mccormick_constraints = mc,
    )
end

"""
    add_bilinear_approx!(config::HybSConfig, container, ::Type{C}, x_var, y_var, x_bounds, y_bounds, meta)

Build x² and y² via `add_quadratic_approx!(config.quad_config, ...)`,
build the (x+y) and (x−y) expression containers, build their epigraphs via
`add_quadratic_approx!(EpigraphQuadConfig(...), ...)`, then allocate the
HybS product variable + bound constraints and assemble per cell.
"""
function add_bilinear_approx!(
    config::HybSConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    xsq = add_quadratic_approx!(
        config.quad_config,
        container,
        C,
        x_var,
        x_bounds,
        meta * "_x",
    )
    ysq = add_quadratic_approx!(
        config.quad_config,
        container,
        C,
        y_var,
        y_bounds,
        meta * "_y",
    )
    return _add_hybs_adapter!(
        container, C, config, x_var, y_var, xsq, ysq, x_bounds, y_bounds, meta,
    )
end

"""
    add_bilinear_approx!(config::HybSConfig, container, ::Type{C}, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed-form: accepts already-built `xsq` ≈ x² and `ysq` ≈ y² 2D
expression containers; builds the HybS pieces on top.
"""
function add_bilinear_approx!(
    config::HybSConfig,
    container::OptimizationContainer,
    ::Type{C},
    xsq,
    ysq,
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    return _add_hybs_adapter!(
        container, C, config, x_var, y_var, xsq, ysq, x_bounds, y_bounds, meta,
    )
end

# Allocate the HybS-specific containers (sum/diff exprs, two epigraphs, z var,
# bounds, approximation, optional McCormick) and loop (name, t) to assemble.
function _add_hybs_adapter!(
    container::OptimizationContainer, ::Type{C}, config::HybSConfig,
    x_var, y_var, xsq, ysq, x_bounds, y_bounds, meta,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    @assert length(name_axis) == length(x_bounds)
    @assert length(name_axis) == length(y_bounds)
    model = get_jump_model(container)

    p1_target = add_expression_container!(
        container, VariableSumExpression, C, name_axis, time_axis;
        meta = meta * "_plus",
    )
    p2_target = add_expression_container!(
        container, VariableDifferenceExpression, C, name_axis, time_axis;
        meta = meta * "_diff",
    )
    for (i, name) in enumerate(name_axis)
        for t in time_axis
            p1_target[name, t] = x_var[name, t] + y_var[name, t]
            p2_target[name, t] = x_var[name, t] - y_var[name, t]
        end
    end
    p1_bounds = [
        (min = x_bounds[i].min + y_bounds[i].min,
            max = x_bounds[i].max + y_bounds[i].max)
        for i in eachindex(x_bounds)
    ]
    p2_bounds = [
        (min = x_bounds[i].min - y_bounds[i].max,
            max = x_bounds[i].max - y_bounds[i].min)
        for i in eachindex(x_bounds)
    ]

    epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
    zp1 = add_quadratic_approx!(epi_cfg, container, C, p1_target, p1_bounds, meta * "_plus")
    zp2 = add_quadratic_approx!(epi_cfg, container, C, p2_target, p2_bounds, meta * "_diff")

    z_target = add_variable_container!(
        container, BilinearProductVariable, C, name_axis, time_axis; meta,
    )
    bound_target = add_constraints_container!(
        container, HybSBoundConstraint, C, name_axis, 1:2, time_axis; meta,
    )
    approx_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )
    mc_target = if config.add_mccormick
        add_constraints_container!(
        container, McCormickConstraint, C, name_axis, 1:4, time_axis; meta,
    )
    else
        nothing
    end

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        ymn, ymx = y_bounds[i].min, y_bounds[i].max
        z_lo = min(xmn * ymn, xmn * ymx, xmx * ymn, xmx * ymx)
        z_hi = max(xmn * ymn, xmn * ymx, xmx * ymn, xmx * ymx)
        for t in time_axis
            z_v = JuMP.@variable(
                model, lower_bound = z_lo, upper_bound = z_hi,
                base_name = "HybSProduct",
            )
            z_target[name, t] = z_v
            bound_target[name, 1, t] = JuMP.@constraint(
                model, z_v >= 0.5 * (zp1[name, t] - xsq[name, t] - ysq[name, t]),
            )
            bound_target[name, 2, t] = JuMP.@constraint(
                model, z_v <= 0.5 * (xsq[name, t] + ysq[name, t] - zp2[name, t]),
            )
            approx_target[name, t] = JuMP.@expression(model, 1.0 * z_v)
            if config.add_mccormick
                r = build_mccormick_envelope(
                    model, x_var[name, t], y_var[name, t], z_v,
                    xmn, xmx, ymn, ymx,
                )
                for (k, ref) in enumerate(r)
                    mc_target[name, k, t] = ref
                end
            end
        end
    end
    return approx_target
end
