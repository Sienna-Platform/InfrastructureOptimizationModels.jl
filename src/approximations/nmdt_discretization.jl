# Shared NMDT machinery used by both nmdt_quadratic and nmdt_bilinear.
#
# NMDT (Normalized Multiparametric Disaggregation Technique) discretizes a
# normalized variable xh ∈ [0,1] as xh = Σᵢ 2^{−i}·β_i + δ, with β_i ∈ {0,1}
# binary digits and δ ∈ [0, 2^{−L}] a residual. The discretization is then
# combined with another normalized variable (for bilinear products) or with
# itself (for quadratic) via McCormick-linearized binary-continuous products.

# --- Container key types ---

"Binary discretization variables β_i ∈ {0,1} in the NMDT decomposition of xh."
struct NMDTBinaryVariable <: VariableType end
"Residual variable δ ∈ [0, 2^{−L}] capturing the NMDT discretization error."
struct NMDTResidualVariable <: VariableType end
"McCormick linearization variables u_i ≈ β_i·y in NMDT binary-continuous products."
struct NMDTBinaryContinuousProductVariable <: VariableType end
"Variable z ≈ δ · y linearizing the residual-continuous product in NMDT."
struct NMDTResidualProductVariable <: VariableType end

"Expression container for the NMDT binary discretization: Σ 2^{−i}·β_i + δ ≈ xh."
struct NMDTDiscretizationExpression <: ExpressionType end
"Expression container for the NMDT binary-continuous product: Σ 2^{−i}·u_i ≈ β·y."
struct NMDTBinaryContinuousProductExpression <: ExpressionType end

"Constraint enforcing xh = Σ 2^{−i}·β_i + δ in the NMDT discretization."
struct NMDTEDiscretizationConstraint <: ConstraintType end
"Epigraph lower-bound tightening constraint on the NMDT quadratic result."
struct NMDTTightenConstraint <: ConstraintType end

# --- Scalar build helpers (pure JuMP) ---

"""
    build_discretization(model, x, x_min, x_max, depth)

Scalar form: build the NMDT binary discretization of the normalized
variable xh = (x − x_min)/(x_max − x_min) for a single cell. Creates
`depth` binary variables β_i and one residual δ ∈ [0, 2^{−depth}],
enforcing xh = Σ 2^{−i}·β_i + δ.

Returns `(; norm_expr, beta_var, delta_var, disc_constraint, disc_expression)`.
"""
function build_discretization(
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
)
    @assert depth >= 1
    @assert x_max > x_min
    norm_expr = (x - x_min) / (x_max - x_min)
    beta_var = JuMP.@variable(
        model, [i = 1:depth], binary = true, base_name = "NMDTBinary",
    )
    delta_var = JuMP.@variable(
        model,
        lower_bound = 0.0, upper_bound = 2.0^(-depth),
        base_name = "NMDTResidual",
    )
    disc_expr = JuMP.@expression(
        model, sum(2.0^(-i) * beta_var[i] for i in 1:depth) + delta_var,
    )
    disc_con = JuMP.@constraint(model, norm_expr == disc_expr)
    return (;
        norm_expr,
        beta_var,
        delta_var,
        disc_constraint = disc_con,
        disc_expression = disc_expr,
    )
end

"""
    build_binary_continuous_product(model, beta_var, cont_var, cont_min, cont_max, depth; tighten=false)

Scalar form: build Σᵢ 2^{−i}·u_i ≈ β·y for a single cell. `beta_var` is a
1D `DenseAxisArray` over `1:depth` (binary), `cont_var` is a JuMP scalar.

Returns `(; u_var, mccormick_lower, mccormick_upper, result_expression)` where
`mccormick_lower` is `nothing` when `tighten = true`.
"""
function build_binary_continuous_product(
    model::JuMP.Model,
    beta_var::AbstractVector,
    cont_var::JuMP.AbstractJuMPScalar,
    cont_min::Float64,
    cont_max::Float64,
    depth::Int;
    tighten::Bool = false,
)
    u_var = JuMP.@variable(
        model, [i = 1:depth],
        lower_bound = cont_min, upper_bound = cont_max,
        base_name = "NMDTBinContProd",
    )
    upper_1 = JuMP.@constraint(
        model, [i = 1:depth], u_var[i] <= cont_max * beta_var[i],
    )
    upper_2 = JuMP.@constraint(
        model, [i = 1:depth],
        u_var[i] <= cont_min * beta_var[i] + cont_var - cont_min,
    )
    mccormick_lower = if tighten
        nothing
    else
        lower_1 = JuMP.@constraint(
            model, [i = 1:depth], u_var[i] >= cont_min * beta_var[i],
        )
        lower_2 = JuMP.@constraint(
            model, [i = 1:depth],
            u_var[i] >= cont_max * beta_var[i] + cont_var - cont_max,
        )
        (c1 = lower_1, c2 = lower_2)
    end
    result_expr = JuMP.@expression(
        model, sum(2.0^(-i) * u_var[i] for i in 1:depth),
    )
    return (;
        u_var,
        mccormick_lower,
        mccormick_upper = (c1 = upper_1, c2 = upper_2),
        result_expression = result_expr,
    )
end

"""
    build_residual_product(model, delta_var, cont_var, cont_max, depth; tighten=false)

Scalar form: build z ≈ δ·y for a single cell, where δ ∈ [0, 2^{−depth}]
and y ∈ [0, cont_max], using McCormick envelopes. When `tighten`, only the
upper envelopes are added.

Returns `(; z_var, mccormick_constraints)` where `mccormick_constraints` is
the NamedTuple returned by `build_mccormick_upper` (`tighten = true`) or
`build_mccormick_envelope` (`tighten = false`).
"""
function build_residual_product(
    model::JuMP.Model,
    delta_var::JuMP.AbstractJuMPScalar,
    cont_var::JuMP.AbstractJuMPScalar,
    cont_max::Float64,
    depth::Int;
    tighten::Bool = false,
)
    delta_max = 2.0^(-depth)
    z_var = JuMP.@variable(
        model,
        lower_bound = 0.0, upper_bound = delta_max * cont_max,
        base_name = "NMDTResidualProduct",
    )
    mc = if tighten
        build_mccormick_upper(
            model, delta_var, cont_var, z_var, 0.0, delta_max, 0.0, cont_max,
        )
    else
        build_mccormick_envelope(
            model, delta_var, cont_var, z_var, 0.0, delta_max, 0.0, cont_max,
        )
    end
    return (; z_var, mccormick_constraints = mc)
end

"""
    build_assembled_product(model, terms, dz, xh_expr, yh_expr, x_min, x_max, y_min, y_max)

Scalar form: affine reassembly of x·y from normalized NMDT pieces:
    x·y = lx·ly·zh + lx·y_min·xh + ly·x_min·yh + x_min·y_min
where `zh = sum(terms) + dz`. `terms` is a list of scalar AffExpr values.
"""
function build_assembled_product(
    model::JuMP.Model,
    terms::AbstractVector,
    dz::JuMP.AbstractJuMPScalar,
    xh_expr,
    yh_expr,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    lx = x_max - x_min
    ly = y_max - y_min
    return JuMP.@expression(
        model,
        lx * ly * (sum(terms) + dz) +
        lx * y_min * xh_expr +
        ly * x_min * yh_expr +
        x_min * y_min,
    )
end

"""
    build_assembled_dnmdt(model, bx_yh, by_dx, by_xh, bx_dy, dz, xh_expr, yh_expr, x_min, x_max, y_min, y_max; lambda)

Scalar form: convex combination of two NMDT estimates of x·y at one cell.
"""
function build_assembled_dnmdt(
    model::JuMP.Model,
    bx_yh::JuMP.AbstractJuMPScalar,
    by_dx::JuMP.AbstractJuMPScalar,
    by_xh::JuMP.AbstractJuMPScalar,
    bx_dy::JuMP.AbstractJuMPScalar,
    dz::JuMP.AbstractJuMPScalar,
    xh_expr,
    yh_expr,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64;
    lambda::Float64 = DNMDT_LAMBDA,
)
    z1 = build_assembled_product(
        model, [bx_yh, by_dx], dz, xh_expr, yh_expr,
        x_min, x_max, y_min, y_max,
    )
    z2 = build_assembled_product(
        model, [by_xh, bx_dy], dz, yh_expr, xh_expr,
        y_min, y_max, x_min, x_max,
    )
    return JuMP.@expression(model, lambda * z1 + (1.0 - lambda) * z2)
end

# --- IOM allocation + per-cell write helpers (used by NMDT quad/bilinear adapters) ---

function _alloc_discretization_targets!(
    container::OptimizationContainer, ::Type{C}, name_axis, time_axis, depth::Int, meta,
) where {C <: IS.InfrastructureSystemsComponent}
    return (
        norm = add_expression_container!(
            container, NormedVariableExpression, C, name_axis, time_axis; meta,
        ),
        beta = add_variable_container!(
            container, NMDTBinaryVariable, C, name_axis, 1:depth, time_axis; meta,
        ),
        delta = add_variable_container!(
            container, NMDTResidualVariable, C, name_axis, time_axis; meta,
        ),
        disc_expr = add_expression_container!(
            container, NMDTDiscretizationExpression, C, name_axis, time_axis; meta,
        ),
        disc_cons = add_constraints_container!(
            container, NMDTEDiscretizationConstraint, C, name_axis, time_axis; meta,
        ),
    )
end

function _write_discretization_cell!(targets, name, t, r, depth::Int)
    targets.norm[name, t] = r.norm_expr
    for i in 1:depth
        targets.beta[name, i, t] = r.beta_var[i]
    end
    targets.delta[name, t] = r.delta_var
    targets.disc_expr[name, t] = r.disc_expression
    targets.disc_cons[name, t] = r.disc_constraint
    return
end

function _alloc_binary_continuous_product_targets!(
    container::OptimizationContainer, ::Type{C}, name_axis, time_axis, depth::Int, meta;
    tighten::Bool,
) where {C <: IS.InfrastructureSystemsComponent}
    u_target = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable, C,
        name_axis, 1:depth, time_axis; meta,
    )
    mc_meta = meta * "_bc"
    mc_upper_target = add_constraints_container!(
        container, McCormickUpperConstraint, C,
        name_axis, 1:depth, 1:2, time_axis; meta = mc_meta,
    )
    mc_lower_target = tighten ? nothing : add_constraints_container!(
        container, McCormickUpperConstraint, C,
        name_axis, 1:depth, 1:2, time_axis; meta = mc_meta * "_lb",
    )
    result_expr_target = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression, C,
        name_axis, time_axis; meta,
    )
    return (
        u = u_target,
        mc_upper = mc_upper_target,
        mc_lower = mc_lower_target,
        result_expr = result_expr_target,
    )
end

function _write_binary_continuous_product_cell!(targets, name, t, r, depth::Int)
    for i in 1:depth
        targets.u[name, i, t] = r.u_var[i]
        targets.mc_upper[name, i, 1, t] = r.mccormick_upper.c1[i]
        targets.mc_upper[name, i, 2, t] = r.mccormick_upper.c2[i]
        if targets.mc_lower !== nothing
            targets.mc_lower[name, i, 1, t] = r.mccormick_lower.c1[i]
            targets.mc_lower[name, i, 2, t] = r.mccormick_lower.c2[i]
        end
    end
    targets.result_expr[name, t] = r.result_expression
    return
end

function _alloc_residual_product_targets!(
    container::OptimizationContainer, ::Type{C}, name_axis, time_axis, meta;
    tighten::Bool,
) where {C <: IS.InfrastructureSystemsComponent}
    z_target = add_variable_container!(
        container, NMDTResidualProductVariable, C, name_axis, time_axis; meta,
    )
    res_meta = meta * "_res"
    mc_target = if tighten
        add_constraints_container!(
            container, McCormickUpperConstraint, C,
            name_axis, 1:2, time_axis; meta = res_meta,
        )
    else
        add_constraints_container!(
            container, McCormickConstraint, C,
            name_axis, 1:4, time_axis; meta = res_meta,
        )
    end
    return (z = z_target, mc = mc_target, tighten = tighten)
end

function _write_residual_product_cell!(targets, name, t, r)
    targets.z[name, t] = r.z_var
    mc = r.mccormick_constraints
    if targets.tighten
        targets.mc[name, 1, t] = mc.upper_1
        targets.mc[name, 2, t] = mc.upper_2
    else
        targets.mc[name, 1, t] = mc.upper_1
        targets.mc[name, 2, t] = mc.upper_2
        targets.mc[name, 3, t] = mc.lower_1
        targets.mc[name, 4, t] = mc.lower_2
    end
    return
end
