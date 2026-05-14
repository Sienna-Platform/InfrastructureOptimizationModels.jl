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

# --- NMDTDiscretization struct ---

"""
NMDT discretization scaffolding for a single normalized variable xh ∈ [0,1].

Holds the affine expression for the normalized variable, the binary digit
variables β_i (one per level of depth), and the residual δ.
"""
struct NMDTDiscretization{
    NE <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
    BV <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    DV <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 2},
    DC <: JuMP.Containers.DenseAxisArray,
    DE <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
}
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

`mccormick_lower` is `nothing` when `tighten = true` (the caller supplies a
tighter bound elsewhere).
"""
struct NMDTBinaryContinuousProduct{
    UV <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    MCL <: Union{Nothing, NTuple{2, <:JuMP.Containers.DenseAxisArray}},
    MCU <: NTuple{2, <:JuMP.Containers.DenseAxisArray},
    RE <: JuMP.Containers.DenseAxisArray{JuMP.AffExpr, 2},
}
    u_var::UV
    mccormick_lower::MCL
    mccormick_upper::MCU
    result_expression::RE
end

"""
    build_binary_continuous_product(model, beta_var, cont_var, cont_min, cont_max, depth; tighten=false)

Build the depth-level binary-continuous product Σᵢ 2^{−i}·u_i ≈ β·y.

For each (name, i, t), creates an auxiliary u_i with bounds [cont_min, cont_max]
and adds McCormick envelope inequalities on (β_i, y, u_i). If
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
    # McCormick envelopes for u[name, i, t] ≈ cont_var[name, t] · beta[name, i, t],
    # with cont_var ∈ [cont_min, cont_max] and beta ∈ {0, 1}:
    #   c1 (lower): u ≥ cont_min · beta
    #   c2 (lower): u ≥ cont_max · beta + cont_var − cont_max
    #   c3 (upper): u ≤ cont_max · beta
    #   c4 (upper): u ≤ cont_min · beta + cont_var − cont_min
    upper_1 = JuMP.@constraint(
        model,
        [name = name_axis, i = 1:depth, t = time_axis],
        u_var[name, i, t] <= cont_max * beta_var[name, i, t],
    )
    upper_2 = JuMP.@constraint(
        model,
        [name = name_axis, i = 1:depth, t = time_axis],
        u_var[name, i, t] <=
        cont_min * beta_var[name, i, t] + cont_var[name, t] - cont_min,
    )
    mccormick_lower = if tighten
        nothing
    else
        lower_1 = JuMP.@constraint(
            model,
            [name = name_axis, i = 1:depth, t = time_axis],
            u_var[name, i, t] >= cont_min * beta_var[name, i, t],
        )
        lower_2 = JuMP.@constraint(
            model,
            [name = name_axis, i = 1:depth, t = time_axis],
            u_var[name, i, t] >=
            cont_max * beta_var[name, i, t] + cont_var[name, t] - cont_max,
        )
        (lower_1, lower_2)
    end
    result_expr = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        sum(2.0^(-i) * u_var[name, i, t] for i in 1:depth)
    )
    return NMDTBinaryContinuousProduct(
        u_var, mccormick_lower, (upper_1, upper_2), result_expr,
    )
end

"""
Result of the residual-continuous product step z ≈ δ·y. Returned by
`build_residual_product`.
"""
struct NMDTResidualProduct{
    ZV <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 2},
    MC <: NamedTuple,
}
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
    # Bounds for the vectorized envelope: δ ∈ [0, delta_max], cont ∈ [0, cont_max].
    delta_bounds = fill((min = 0.0, max = delta_max), length(name_axis))
    cont_bounds = fill((min = 0.0, max = cont_max), length(name_axis))
    mc = build_mccormick_envelope(
        model,
        delta_var,
        cont_var,
        z_var,
        delta_bounds,
        cont_bounds;
        lower_bounds = !tighten,
    )
    return NMDTResidualProduct(z_var, mc)
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
        container, NormedVariableExpression, C, name_axis, time_axis; meta,
    )
    norm_target.data .= disc.norm_expr.data

    beta_target = add_variable_container!(
        container, NMDTBinaryVariable, C, name_axis, depth_axis, time_axis; meta,
    )
    beta_target.data .= disc.beta_var.data

    delta_target = add_variable_container!(
        container, NMDTResidualVariable, C, name_axis, time_axis; meta,
    )
    delta_target.data .= disc.delta_var.data

    disc_expr_target = add_expression_container!(
        container, NMDTDiscretizationExpression, C, name_axis, time_axis; meta,
    )
    disc_expr_target.data .= disc.disc_expression.data

    disc_cons_target = add_constraints_container!(
        container, NMDTEDiscretizationConstraint, C, name_axis, time_axis; meta,
    )
    disc_cons_target.data .= disc.disc_constraints.data
    return
end

"""
    register_binary_continuous_product!(container, ::Type{C}, product, meta)

Register the auxiliary u variables, McCormick constraints (split into
lower/upper sides under `McCormickLowerConstraint`/`McCormickUpperConstraint`),
and the weighted-sum expression of an NMDT binary-continuous product step.
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
        name_axis,
        depth_axis,
        time_axis;
        meta,
    )
    u_target.data .= product.u_var.data

    # Suffix the McCormick meta so the binary-continuous product's envelope
    # doesn't collide with a sibling residual product's envelope under the
    # same NMDT approximation's `meta`.
    _register_mccormick_depth_side!(
        container, C, McCormickUpperConstraint, product.mccormick_upper, meta * "_bc",
    )
    _register_mccormick_depth_side!(
        container, C, McCormickLowerConstraint, product.mccormick_lower, meta * "_bc",
    )

    expr_target = add_expression_container!(
        container,
        NMDTBinaryContinuousProductExpression,
        C,
        name_axis,
        time_axis;
        meta,
    )
    expr_target.data .= product.result_expression.data
    return
end

# Register one side (lower or upper) of an NMDT binary-continuous product's
# McCormick envelope. Each side is a pair of 3D `(name, depth, t)` constraint
# containers; we stack them into a 4D `(name, depth, k=1:2, t)` container.
function _register_mccormick_depth_side!(
    container::OptimizationContainer,
    ::Type{C},
    ::Type{K},
    cons::Tuple{<:JuMP.Containers.DenseAxisArray, <:JuMP.Containers.DenseAxisArray},
    meta::String,
) where {
    C <: IS.InfrastructureSystemsComponent,
    K <: ConstraintType,
}
    c1, c2 = cons
    name_axis = axes(c1, 1)
    depth_axis = axes(c1, 2)
    time_axis = axes(c1, 3)
    target = add_constraints_container!(
        container, K, C, name_axis, depth_axis, 1:2, time_axis; meta,
    )
    @views target.data[:, :, 1, :] .= c1.data
    @views target.data[:, :, 2, :] .= c2.data
    return
end

# No-op when the lower side wasn't built.
_register_mccormick_depth_side!(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Type{<:ConstraintType},
    ::Nothing,
    ::String,
) = nothing

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
        container, NMDTResidualProductVariable, C, name_axis, time_axis; meta,
    )
    z_target.data .= product.z_var.data

    register_mccormick_envelope!(container, C, product.mccormick_constraints, meta)
    return
end
