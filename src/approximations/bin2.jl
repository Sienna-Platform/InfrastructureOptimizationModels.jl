# Bin2 separable approximation of bilinear products z = x·y.
# Uses the identity x·y = ½·((x+y)² − x² − y²).
# Composes a quadratic approximation (chosen via `quad_config`) for x², y²,
# and (x+y)². Optionally adds reformulated McCormick cuts to tighten the LP
# relaxation in terms of the three quadratic approximations.

"""
Config for Bin2 bilinear approximation using z = ½·((x+y)² − x² − y²).

# Fields
- `quad_config::QuadraticApproxConfig`: quadratic method used for x², y², and (x+y)².
- `add_mccormick::Bool`: whether to add reformulated McCormick cuts (default true).
"""
struct Bin2Config{QC <: QuadraticApproxConfig} <: BilinearApproxConfig
    quad_config::QC
    add_mccormick::Bool
end
function Bin2Config(quad_config::QuadraticApproxConfig)
    return Bin2Config(quad_config, true)
end

"""
    build_bilinear_approx(config::Bin2Config, model, x, y, x_min, x_max, y_min, y_max)

Scalar form: build x², y², (x+y)² via the chosen quadratic method, combine
via z = ½·(psq − xsq − ysq). If `config.add_mccormick`, also build the
four reformulated McCormick cuts.

Returns `(; approximation, xsq, ysq, psq, sum_expression, mccormick_constraints)`
where `mccormick_constraints` is `nothing` or a NamedTuple `(c1, c2, c3, c4)`.
"""
function build_bilinear_approx(
    config::Bin2Config,
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
    p_expr = JuMP.@expression(model, x + y)
    psq = build_quadratic_approx(
        config.quad_config, model, p_expr, x_min + y_min, x_max + y_max,
    )
    approximation = JuMP.@expression(
        model,
        0.5 * (psq.approximation - xsq.approximation - ysq.approximation),
    )
    mc = config.add_mccormick ?
        build_reformulated_mccormick(
            model, x, y,
            psq.approximation, xsq.approximation, ysq.approximation,
            x_min, x_max, y_min, y_max,
        ) :
        nothing
    return (;
        approximation,
        xsq,
        ysq,
        psq,
        sum_expression = p_expr,
        mccormick_constraints = mc,
    )
end

"""
    add_bilinear_approx!(config::Bin2Config, container, ::Type{C}, x_var, y_var, x_bounds, y_bounds, meta)

Build x² and y² via `add_quadratic_approx!(config.quad_config, ...)`,
build the (x+y) expression container and its psq via the same quad
adapter, then assemble z = ½·(psq − xsq − ysq) and (optionally) the
reformulated McCormick cuts.
"""
function add_bilinear_approx!(
    config::Bin2Config,
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
    @assert length(name_axis) == length(x_bounds)
    @assert length(name_axis) == length(y_bounds)
    model = get_jump_model(container)

    p_target = add_expression_container!(
        container, VariableSumExpression, C, name_axis, time_axis;
        meta = meta * "_plus",
    )
    for (i, name) in enumerate(name_axis)
        for t in time_axis
            p_target[name, t] = x_var[name, t] + y_var[name, t]
        end
    end
    p_bounds = [
        (min = x_bounds[i].min + y_bounds[i].min,
            max = x_bounds[i].max + y_bounds[i].max)
        for i in eachindex(x_bounds)
    ]

    xsq = add_quadratic_approx!(config.quad_config, container, C, x_var, x_bounds, meta * "_x")
    ysq = add_quadratic_approx!(config.quad_config, container, C, y_var, y_bounds, meta * "_y")
    psq = add_quadratic_approx!(
        config.quad_config, container, C, p_target, p_bounds, meta * "_plus",
    )

    return _bin2_assemble_and_mccormick!(
        container, C, name_axis, time_axis, model,
        x_var, y_var, xsq, ysq, psq, x_bounds, y_bounds, meta;
        add_mccormick = config.add_mccormick,
    )
end

"""
    add_bilinear_approx!(config::Bin2Config, container, ::Type{C}, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Precomputed-form: accepts already-built `xsq` ≈ x² and `ysq` ≈ y² 2D
expression containers (rather than rebuilding them). Builds only the
(x+y)² approximation on top, the Bin2 assembly, and the optional cuts.
"""
function add_bilinear_approx!(
    config::Bin2Config,
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
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    @assert length(name_axis) == length(x_bounds)
    @assert length(name_axis) == length(y_bounds)
    model = get_jump_model(container)

    p_target = add_expression_container!(
        container, VariableSumExpression, C, name_axis, time_axis;
        meta = meta * "_plus",
    )
    for (i, name) in enumerate(name_axis)
        for t in time_axis
            p_target[name, t] = x_var[name, t] + y_var[name, t]
        end
    end
    p_bounds = [
        (min = x_bounds[i].min + y_bounds[i].min,
            max = x_bounds[i].max + y_bounds[i].max)
        for i in eachindex(x_bounds)
    ]
    psq = add_quadratic_approx!(
        config.quad_config, container, C, p_target, p_bounds, meta * "_plus",
    )

    return _bin2_assemble_and_mccormick!(
        container, C, name_axis, time_axis, model,
        x_var, y_var, xsq, ysq, psq, x_bounds, y_bounds, meta;
        add_mccormick = config.add_mccormick,
    )
end

# Allocate the BilinearProductExpression result + optional ReformulatedMcCormick
# container, then loop (name, t) to assemble z and the McCormick cuts.
function _bin2_assemble_and_mccormick!(
    container, ::Type{C}, name_axis, time_axis, model,
    x_var, y_var, xsq, ysq, psq, x_bounds, y_bounds, meta;
    add_mccormick::Bool,
) where {C <: IS.InfrastructureSystemsComponent}
    approx_target = add_expression_container!(
        container, BilinearProductExpression, C, name_axis, time_axis; meta,
    )
    mc_target = add_mccormick ?
        add_constraints_container!(
            container, ReformulatedMcCormickConstraint, C,
            name_axis, 1:4, time_axis; meta,
        ) : nothing

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        ymn, ymx = y_bounds[i].min, y_bounds[i].max
        for t in time_axis
            approx_target[name, t] =
                0.5 * (psq[name, t] - xsq[name, t] - ysq[name, t])
            if add_mccormick
                r = build_reformulated_mccormick(
                    model,
                    x_var[name, t], y_var[name, t],
                    psq[name, t], xsq[name, t], ysq[name, t],
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
