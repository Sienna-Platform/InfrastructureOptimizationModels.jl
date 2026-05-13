# Shared NMDT machinery used by both nmdt_quadratic and nmdt_bilinear.
#
# NMDT (Normalized Multiparametric Disaggregation Technique) discretizes a
# normalized variable xh ∈ [0,1] as xh = Σᵢ 2^{−i}·β_i + δ, with β_i ∈ {0,1}
# binary digits and δ ∈ [0, 2^{−L}] a residual. The discretization is then
# combined with another normalized variable (for bilinear products) or with
# itself (for quadratic) via McCormick-linearized binary-continuous products.
#
# This file provides:
#   - The container key types
#   - The NMDTDiscretization struct (intermediate scaffolding type)
#   - Pure-JuMP build helpers for each NMDT building block
#   - Per-piece register_* helpers used by the top-level method's
#     register_in_container! implementation

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
"McCormick envelope constraints for binary-continuous products u_i ≈ β_i·y in NMDT."
struct NMDTBinaryContinuousProductConstraint <: ConstraintType end
"Epigraph lower-bound tightening constraint on the NMDT quadratic result."
struct NMDTTightenConstraint <: ConstraintType end

# --- NMDTDiscretization struct ---

"""
NMDT discretization scaffolding for a single normalized variable xh ∈ [0,1].

Holds the affine expression for the normalized variable, the binary digit
variables β_i (one per level of depth), and the residual δ. Constructed by
`build_discretization` and consumed by `build_binary_continuous_product`,
`build_residual_product`, and the NMDT assembly helpers.
"""
struct NMDTDiscretization{NE, BV, DV, DC, DE}
    norm_expr::NE
    beta_var::BV
    delta_var::DV
    disc_constraints::DC
    disc_expression::DE
end

# --- Pure-JuMP build helpers ---

"""
    build_discretization(model, x, bounds, depth) -> NMDTDiscretization

Build the NMDT binary discretization of the normalized variable
xh = (x − x_min)/(x_max − x_min). Creates `depth` binary variables β_i and
one residual δ_h ∈ [0, 2^{−depth}], enforcing xh = Σ 2^{−i}·β_i + δ_h.
"""
function build_discretization(
    model::JuMP.Model,
    x,
    bounds::Vector{MinMax},
    depth::Int,
)
    IS.@assert_op depth >= 1
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(bounds)

    norm_expr = build_normed_variable(model, x, bounds)
    beta_var = JuMP.@variable(
        model,
        [name = name_axis, i = 1:depth, t = time_axis],
        binary = true,
        base_name = "NMDTBinary",
    )
    delta_var = JuMP.@variable(
        model,
        [name = name_axis, t = time_axis],
        lower_bound = 0.0,
        upper_bound = 2.0^(-depth),
        base_name = "NMDTResidual",
    )
    disc_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        sum(2.0^(-i) * beta_var[name, i, t] for i in 1:depth) + delta_var[name, t]
    )
    disc_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        norm_expr[name, t] == disc_expr[name, t],
    )
    return NMDTDiscretization(norm_expr, beta_var, delta_var, disc_cons, disc_expr)
end

"""
Result of a single NMDT binary-continuous product step β_i·y ≈ u_i, weighted
sum into Σ 2^{−i}·u_i. Returned by `build_binary_continuous_product`.
"""
struct NMDTBinaryContinuousProduct{UV, MC, RE}
    u_var::UV
    mccormick_constraints::MC
    result_expression::RE
end

"""
    build_binary_continuous_product(model, beta_var, cont_var, cont_min, cont_max, depth; tighten=false)

Build the depth-level binary-continuous product Σᵢ 2^{−i}·u_i ≈ β·y.

For each (name, i, t), creates an auxiliary u_i with bounds [cont_min, cont_max]
and adds the four McCormick envelope inequalities on (β_i, y, u_i). If
`tighten = true`, the lower-bound McCormick constraints are omitted (the caller
applies a tighter bound elsewhere).
"""
function build_binary_continuous_product(
    model::JuMP.Model,
    beta_var,
    cont_var,
    cont_min::Float64,
    cont_max::Float64,
    depth::Int;
    tighten::Bool = false,
)
    name_axis = axes(beta_var, 1)
    time_axis = axes(beta_var, 3)
    u_var = JuMP.@variable(
        model,
        [name = name_axis, i = 1:depth, t = time_axis],
        lower_bound = cont_min,
        upper_bound = cont_max,
        base_name = "NMDTBinContProd",
    )
    mc_cons = JuMP.Containers.DenseAxisArray{Any}(
        undef, name_axis, 1:depth, 1:4, time_axis,
    )
    for name in name_axis, i in 1:depth, t in time_axis
        c1, c2, c3, c4 = build_mccormick_envelope(
            model,
            cont_var[name, t],
            beta_var[name, i, t],
            u_var[name, i, t],
            cont_min,
            cont_max,
            0.0,
            1.0;
            lower_bounds = !tighten,
        )
        mc_cons[name, i, 1, t] = c1
        mc_cons[name, i, 2, t] = c2
        mc_cons[name, i, 3, t] = c3
        mc_cons[name, i, 4, t] = c4
    end
    result_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        sum(2.0^(-i) * u_var[name, i, t] for i in 1:depth)
    )
    return NMDTBinaryContinuousProduct(u_var, mc_cons, result_expr)
end

"""
Result of the residual-continuous product step z ≈ δ·y. Returned by
`build_residual_product`.
"""
struct NMDTResidualProduct{ZV, MC}
    z_var::ZV
    mccormick_constraints::MC
end

"""
    build_residual_product(model, delta_var, cont_var, cont_max, depth; tighten=false)

Build a single auxiliary z ≈ δ·y with McCormick envelopes on
(δ ∈ [0, 2^{−depth}], y ∈ [0, cont_max]). Lower bounds on the McCormick
envelope are omitted when `tighten = true`.
"""
function build_residual_product(
    model::JuMP.Model,
    delta_var,
    cont_var,
    cont_max::Float64,
    depth::Int;
    tighten::Bool = false,
)
    name_axis = axes(delta_var, 1)
    time_axis = axes(delta_var, 2)
    delta_max = 2.0^(-depth)
    z_var = JuMP.@variable(
        model,
        [name = name_axis, t = time_axis],
        lower_bound = 0.0,
        upper_bound = delta_max * cont_max,
        base_name = "NMDTResidualProduct",
    )
    mc_cons = JuMP.Containers.DenseAxisArray{Any}(undef, name_axis, 1:4, time_axis)
    for name in name_axis, t in time_axis
        c1, c2, c3, c4 = build_mccormick_envelope(
            model,
            delta_var[name, t],
            cont_var[name, t],
            z_var[name, t],
            0.0,
            delta_max,
            0.0,
            cont_max;
            lower_bounds = !tighten,
        )
        mc_cons[name, 1, t] = c1
        mc_cons[name, 2, t] = c2
        mc_cons[name, 3, t] = c3
        mc_cons[name, 4, t] = c4
    end
    return NMDTResidualProduct(z_var, mc_cons)
end

"""
    build_assembled_product(model, terms, dz, xh_expr, yh_expr, x_bounds, y_bounds)

Affine reassembly of the bilinear product x·y from normalized NMDT pieces.

For each (name, t):
    x·y = lx·ly·zh + lx·y_min·xh + ly·x_min·yh + x_min·y_min
where `zh = sum(term[name, t] for term in terms) + dz[name, t]`.
"""
function build_assembled_product(
    model::JuMP.Model,
    terms,
    dz,
    xh_expr,
    yh_expr,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
)
    name_axis = axes(xh_expr, 1)
    time_axis = axes(xh_expr, 2)
    IS.@assert_op length(name_axis) == length(x_bounds)
    IS.@assert_op length(name_axis) == length(y_bounds)
    lx = JuMP.Containers.DenseAxisArray(
        [b.max - b.min for b in x_bounds],
        name_axis,
    )
    ly = JuMP.Containers.DenseAxisArray(
        [b.max - b.min for b in y_bounds],
        name_axis,
    )
    x_min = JuMP.Containers.DenseAxisArray(
        [b.min for b in x_bounds],
        name_axis,
    )
    y_min = JuMP.Containers.DenseAxisArray(
        [b.min for b in y_bounds],
        name_axis,
    )
    return JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        lx[name] * ly[name] *
        (sum(term[name, t] for term in terms) + dz[name, t]) +
        lx[name] * y_min[name] * xh_expr[name, t] +
        ly[name] * x_min[name] * yh_expr[name, t] +
        x_min[name] * y_min[name]
    )
end

"""
    build_assembled_dnmdt(model, bx_yh, by_dx, by_xh, bx_dy, dz, x_disc, y_disc, x_bounds, y_bounds; lambda)

Convex combination of two NMDT estimates of x·y. Returns the result expression.

`z1` is the (x discretizes, y normalized) estimate and `z2` is the (y discretizes,
x normalized) estimate. The result is `λ·z1 + (1−λ)·z2`. The shared residual
product `dz ≈ δ_x · δ_y` is supplied by the caller.
"""
function build_assembled_dnmdt(
    model::JuMP.Model,
    bx_yh,
    by_dx,
    by_xh,
    bx_dy,
    dz,
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax};
    lambda::Float64 = DNMDT_LAMBDA,
)
    z1 = build_assembled_product(
        model, [bx_yh, by_dx], dz,
        x_disc.norm_expr, y_disc.norm_expr, x_bounds, y_bounds,
    )
    z2 = build_assembled_product(
        model, [by_xh, bx_dy], dz,
        y_disc.norm_expr, x_disc.norm_expr, y_bounds, x_bounds,
    )
    name_axis = axes(z1, 1)
    time_axis = axes(z1, 2)
    return JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        lambda * z1[name, t] + (1.0 - lambda) * z2[name, t]
    )
end

# --- IOM-side register helpers (called from method-specific register_in_container!) ---

"""
    register_discretization!(container, ::Type{C}, disc::NMDTDiscretization, meta)

Register the discretization variables, residual, expression, and constraint
into the optimization container under their respective key types.
"""
function register_discretization!(
    container::OptimizationContainer,
    ::Type{C},
    disc::NMDTDiscretization,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(disc.beta_var, 1)
    depth_axis = axes(disc.beta_var, 2)
    time_axis = axes(disc.beta_var, 3)

    norm_target = add_expression_container!(
        container,
        NormedVariableExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        norm_target[name, t] = disc.norm_expr[name, t]
    end

    beta_target = add_variable_container!(
        container,
        NMDTBinaryVariable,
        C,
        collect(name_axis),
        depth_axis,
        time_axis;
        meta,
    )
    for name in name_axis, i in depth_axis, t in time_axis
        beta_target[name, i, t] = disc.beta_var[name, i, t]
    end

    delta_target = add_variable_container!(
        container,
        NMDTResidualVariable,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        delta_target[name, t] = disc.delta_var[name, t]
    end

    disc_expr_target = add_expression_container!(
        container,
        NMDTDiscretizationExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        disc_expr_target[name, t] = disc.disc_expression[name, t]
    end

    disc_cons_target = add_constraints_container!(
        container,
        NMDTEDiscretizationConstraint,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        disc_cons_target[name, t] = disc.disc_constraints[name, t]
    end
    return
end

"""
    register_binary_continuous_product!(container, ::Type{C}, product, meta)

Register the auxiliary u variables, McCormick constraints, and weighted-sum
expression of an NMDT binary-continuous product step.
"""
function register_binary_continuous_product!(
    container::OptimizationContainer,
    ::Type{C},
    product::NMDTBinaryContinuousProduct,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(product.u_var, 1)
    depth_axis = axes(product.u_var, 2)
    time_axis = axes(product.u_var, 3)

    u_target = add_variable_container!(
        container,
        NMDTBinaryContinuousProductVariable,
        C,
        collect(name_axis),
        depth_axis,
        time_axis;
        meta,
    )
    for name in name_axis, i in depth_axis, t in time_axis
        u_target[name, i, t] = product.u_var[name, i, t]
    end

    cons_target = add_constraints_container!(
        container,
        NMDTBinaryContinuousProductConstraint,
        C,
        collect(name_axis),
        depth_axis,
        1:4,
        time_axis;
        meta,
    )
    for name in name_axis, i in depth_axis, k in 1:4, t in time_axis
        c = product.mccormick_constraints[name, i, k, t]
        c === nothing && continue
        cons_target[name, i, k, t] = c
    end

    expr_target = add_expression_container!(
        container,
        NMDTBinaryContinuousProductExpression,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        expr_target[name, t] = product.result_expression[name, t]
    end
    return
end

"""
    register_residual_product!(container, ::Type{C}, product, meta)

Register the residual product z variable and its McCormick constraints.
"""
function register_residual_product!(
    container::OptimizationContainer,
    ::Type{C},
    product::NMDTResidualProduct,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(product.z_var, 1)
    time_axis = axes(product.z_var, 2)

    z_target = add_variable_container!(
        container,
        NMDTResidualProductVariable,
        C,
        collect(name_axis),
        time_axis;
        meta,
    )
    for name in name_axis, t in time_axis
        z_target[name, t] = product.z_var[name, t]
    end

    register_mccormick_envelope!(
        container, C, product.mccormick_constraints, meta,
    )
    return
end
