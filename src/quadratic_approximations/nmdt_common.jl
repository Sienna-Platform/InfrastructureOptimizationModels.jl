"Binary discretization variables β_i ∈ {0,1} in the NMDT decomposition of xh."
struct NMDTBinaryVariable <: VariableType end
"Residual variable δ ∈ [0, 2^{−L}] capturing the NMDT discretization error."
struct NMDTResidualVariable <: VariableType end
"McCormick linearization variables u_i ≈ β_i · y in NMDT binary-continuous products."
struct NMDTBinaryContinuousProductVariable <: VariableType end
"Variable z ≈ δ · y linearizing the residual-continuous product in NMDT."
struct NMDTResidualProductVariable <: VariableType end

"Expression container for the NMDT binary discretization: Σ 2^{−i}·β_i + δ ≈ xh."
struct NMDTDiscretizationExpression <: ExpressionType end
"Expression container for the NMDT binary-continuous product: Σ 2^{−i}·u_i ≈ β·y."
struct NMDTBinaryContinuousProductExpression <: ExpressionType end
"Expression container for the final NMDT quadratic approximation result."
struct NMDTResultExpression <: ExpressionType end

"Constraint enforcing xh = Σ 2^{−i}·β_i + δ in the NMDT discretization."
struct NMDTEDiscretizationConstraint <: ConstraintType end
"McCormick envelope constraints for binary-continuous products u_i ≈ β_i·y in NMDT."
struct NMDTBinaryContinuousProductConstraint <: ConstraintType end
"Epigraph lower-bound tightening constraint on the NMDT quadratic result."
struct NMDTTightenConstraint <: ConstraintType end

"""
Stores the result of discretizing a normalized variable for use in NMDT products.

Fields:
- `norm_expr`: affine expression for xh = (x − x_min)/(x_max − x_min) ∈ [0,1]
- `beta_var`: binary variables β_i ∈ {0,1} indexed by (name, i, t)
- `delta_var`: residual variables δ ∈ [0, 2^{−depth}] indexed by (name, t)
"""
struct NMDTDiscretization{NE, BV, DV}
    norm_expr::NE
    beta_var::BV
    delta_var::DV
end

"""
    _discretize!(container, C, names, time_steps, x_var, x_min, x_max, depth, meta)

Discretize the normalized variable xh = (x − x_min)/(x_max − x_min) using L binary variables.

Creates L binary variables β₁,…,β_L and one residual δ ∈ [0, 2^{−L}] such that
xh = Σᵢ 2^{−i}·β_i + δ. Enforces this via a `NMDTEDiscretizationConstraint` and
returns an `NMDTDiscretization` struct holding all components.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `depth::Int`: number of binary discretization levels L
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _discretize!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op depth >= 1
    jump_model = get_jump_model(container)

    beta_var = add_variable_container!(
        container,
        NMDTBinaryVariable,
        C,
        names,
        1:depth,
        time_steps;
        meta,
    )
    delta_var = add_variable_container!(
        container,
        NMDTResidualVariable,
        C,
        names,
        time_steps;
        meta,
    )
    disc_expr = add_expression_container!(
        container,
        NMDTDiscretizationExpression,
        C,
        names,
        time_steps;
        meta,
    )
    disc_cons = add_constraints_container!(
        container,
        NMDTEDiscretizationConstraint,
        C,
        names,
        time_steps;
        meta,
    )

    xh_expr = _normed_variable!(
        container, C, names, time_steps,
        x_var, x_min, x_max, meta,
    )

    for name in names, t in time_steps
        disc = disc_expr[name, t] = JuMP.AffExpr(0.0)
        for i in 1:depth
            beta =
                beta_var[name, i, t] = JuMP.@variable(
                    jump_model,
                    base_name = "NMDTBinary_$(C)_{$(name), $(i), $(t)}",
                    binary = true
                )
            add_proportional_to_jump_expression!(disc, beta, 2.0^(-i))
        end
        delta =
            delta_var[name, t] = JuMP.@variable(
                jump_model,
                base_name = "NMDTResidual_$(C)_{$(name), $(t)}",
                lower_bound = 0.0,
                upper_bound = 2.0^(-depth)
            )
        add_proportional_to_jump_expression!(disc, delta, 1.0)
        disc_cons[name, t] = JuMP.@constraint(
            jump_model,
            xh_expr[name, t] == disc
        )
    end

    return NMDTDiscretization(xh_expr, beta_var, delta_var)
end

"""
    _binary_continuous_product!(container, C, names, time_steps, bin_disc, cont_var, cont_min, cont_max, depth, meta; tighten)

Linearize each binary-continuous product β_i·y using McCormick envelopes.

For each depth level i, creates a variable u_i ≈ β_i·y with bounds [cont_min, cont_max]
and adds 4 McCormick constraints via `_add_mccormick_envelope!`. Assembles the weighted sum
Σᵢ 2^{−i}·u_i into a `NMDTBinaryContinuousProductExpression`.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `bin_disc`: `NMDTDiscretization` providing β_i variables and depth
- `cont_var`: container of continuous variables y indexed by (name, t)
- `cont_min::Float64`: lower bound of y
- `cont_max::Float64`: upper bound of y
- `depth::Int`: number of binary discretization levels
- `meta::String`: identifier encoding the original variable type being approximated
- `tighten::Bool`: if true, omit McCormick lower bounds (for use when a tighter bound is applied elsewhere)
"""
function _binary_continuous_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    bin_disc,
    cont_var,
    cont_min::Float64,
    cont_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)

    u_var = add_variable_container!(
        container,
        NMDTBinaryContinuousProductVariable,
        C,
        names,
        1:depth,
        time_steps;
        meta,
    )
    u_cons = add_constraints_container!(
        container,
        NMDTBinaryContinuousProductConstraint,
        C,
        names,
        1:depth,
        1:4,
        time_steps;
        meta,
    )
    result_expr = add_expression_container!(
        container,
        NMDTBinaryContinuousProductExpression,
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        for i in 1:depth
            u_i =
                u_var[name, i, t] = JuMP.@variable(
                    jump_model,
                    base_name = "NMDTBinContProd_$(C)_{$(name), $(i), $(t)}",
                    lower_bound = cont_min,
                    upper_bound = cont_max
                )
            _add_mccormick_envelope!(
                jump_model, u_cons, (name, i, t),
                cont_var[name, t], bin_disc.beta_var[name, i, t], u_i,
                cont_min, cont_max, 0.0, 1.0;
                lower_bounds = !tighten,
            )
            add_proportional_to_jump_expression!(result, u_i, 2.0^(-i))
        end
    end

    return result_expr
end

"""
    _tighten_lower_bounds!(container, C, names, time_steps, result_expr, x_disc, epigraph_depth, meta)

Add epigraph lower-bound constraints to tighten an NMDT quadratic approximation.

Computes an epigraph Q^{L1} lower bound on xh²
and adds a `NMDTTightenConstraint` enforcing `result_expr[name,t] ≥ epi_expr[name,t]`
for each (name, t). This improves the lower bound quality of NMDT without adding binaries.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `result_expr`: expression container for the NMDT quadratic result to be tightened
- `x_disc`: `NMDTDiscretization` for x, providing `norm_expr` and `depth`
- `epigraph_depth::Int`: depth for the epigraph Q^{L1} lower-bound approximation
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _tighten_lower_bounds!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    result_expr,
    x_disc,
    epigraph_depth::Int,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)

    epi_expr = _add_quadratic_approx!(
        EpigraphQuadConfig(epigraph_depth),
        container, C, names, time_steps,
        x_disc.norm_expr, 0.0, 1.0, meta * "_epi",
    )
    epi_cons = add_constraints_container!(
        container,
        NMDTTightenConstraint,
        C,
        names,
        time_steps;
        meta,
    )
    for name in names, t in time_steps
        epi_cons[name, t] = JuMP.@constraint(
            jump_model,
            result_expr[name, t] >= epi_expr[name, t],
        )
    end
end

"""
    _residual_product!(container, C, names, time_steps, x_disc, y_var, y_max, meta; tighten)

Linearize the residual-continuous product z ≈ δ·y using McCormick envelopes.

Creates a variable z ∈ [0, 2^{−L}·y_max] for each (name, t) and bounds it with
McCormick constraints on (δ, y) where δ ∈ [0, 2^{−L}]. Stores results in a
`NMDTResidualProductVariable` container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_disc`: `NMDTDiscretization` for x, providing `delta_var` and `depth`
- `y_var`: container of continuous variables y indexed by (name, t)
- `y_max::Float64`: upper bound of y (lower bound assumed 0)
- `meta::String`: identifier encoding the original variable type being approximated
- `tighten::Bool`: if true, omit McCormick lower bounds (default: false)
"""
function _residual_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc,
    y_var,
    y_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    x_max = 2.0^(-depth)
    jump_model = get_jump_model(container)

    z_var = add_variable_container!(
        container,
        NMDTResidualProductVariable,
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        z_var[name, t] = JuMP.@variable(
            jump_model,
            base_name = "NMDTResidualProduct_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = x_max * y_max,
        )
    end

    _add_mccormick_envelope!(
        container, C, names, time_steps,
        x_disc.delta_var, y_var, z_var,
        0.0, x_max, 0.0, y_max,
        meta; lower_bounds = !tighten,
    )

    return z_var
end

"""
    _assemble_product!(container, C, names, time_steps, terms, dz_var, x_disc, y_disc, meta; result_type)

Reconstruct the bilinear product x·y from normalized NMDT components.

Applies the affine rescaling:
```
x·y = lx·ly·zh + lx·y_min·xh + ly·x_min·yh + x_min·y_min
```
where `zh = Σ terms[name,t] + dz_var[name,t]` collects the binary-continuous and
residual product contributions, lx = x_max − x_min, ly = y_max − y_min.

Stores results in an expression container of type `result_type`.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `terms`: iterable of expression containers indexed by (name, t) for the binary-continuous products
- `dz_var`: variable container for the residual product δ·y
- `xh_norm`: normed expression container for x
- `yh_norm`: normed expression container for y
- `x_min::Float64`: original x min
- `x_max::Float64`: original x max
- `y_min::Float64`: original y min
- `y_max::Float64`: original y max
- `meta::String`: identifier encoding the original variable type being approximated
- `result_type`: expression type to store results in (default: `NMDTResultExpression`)
"""
function _assemble_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    terms,
    dz_var,
    xh_expr,
    yh_expr,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    meta::String;
    result_type = NMDTResultExpression,
) where {C <: IS.InfrastructureSystemsComponent}
    lx = x_max - x_min
    ly = y_max - y_min

    result_expr = add_expression_container!(
        container,
        result_type,
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        zh = JuMP.AffExpr(0.0)
        for term in terms
            add_proportional_to_jump_expression!(zh, term[name, t], 1.0)
        end
        add_proportional_to_jump_expression!(zh, dz_var[name, t], 1.0)

        add_proportional_to_jump_expression!(result, zh, lx * ly)
        add_proportional_to_jump_expression!(result, xh_expr[name, t], lx * y_min)
        add_proportional_to_jump_expression!(result, yh_expr[name, t], ly * x_min)
        add_constant_to_jump_expression!(result, x_min * y_min)
    end

    return result_expr
end

"""
    _assemble_product!(container, C, names, time_steps, terms, dz_var, x_disc::NMDTDiscretization, y_disc::NMDTDiscretization, x_min, x_max, y_min, y_max, meta; result_type)

Convenience overload: extracts `norm_expr` from both discretizations and delegates to the core `_assemble_product!`.
"""
function _assemble_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    terms,
    dz_var,
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    meta::String;
    result_type = NMDTResultExpression,
) where {C <: IS.InfrastructureSystemsComponent}
    return _assemble_product!(
        container, C, names, time_steps, terms, dz_var,
        x_disc.norm_expr, y_disc.norm_expr,
        x_min, x_max, y_min, y_max,
        meta; result_type,
    )
end

"""
    _assemble_product!(container, C, names, time_steps, terms, dz_var, x_disc::NMDTDiscretization, yh_expr, x_min, x_max, y_min, y_max, meta; result_type)

Convenience overload: extracts `norm_expr` from x_disc and delegates to the core `_assemble_product!`.
"""
function _assemble_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    terms,
    dz_var,
    x_disc::NMDTDiscretization,
    yh_expr,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    meta::String;
    result_type = NMDTResultExpression,
) where {C <: IS.InfrastructureSystemsComponent}
    return _assemble_product!(
        container, C, names, time_steps, terms, dz_var,
        x_disc.norm_expr, yh_expr,
        x_min, x_max, y_min, y_max,
        meta; result_type,
    )
end

"""
    _assemble_dnmdt!(container, C, names, time_steps, bx_yh_expr, by_dx_expr, by_xh_expr, bx_dy_expr, x_disc, y_disc, meta; lambda, result_type)

Core assembler for the DNMDT bilinear approximation of x·y from pre-computed cross products.

Builds two NMDT product estimates from opposite discretization pairings and combines them:
- z₁ = assemble(bx·yh + by·δx + δx·δy, x_disc, y_disc)
- z₂ = assemble(by·xh + bx·δy + δx·δy, y_disc, x_disc)
- result = λ·z₁ + (1−λ)·z₂

The shared residual product δx·δy is computed internally. Stores results in an expression
container of type `result_type`.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `bx_yh_expr`: expression for β_x·yh binary-continuous products
- `by_dx_expr`: expression for β_y·δx binary-continuous products
- `by_xh_expr`: expression for β_y·xh binary-continuous products
- `bx_dy_expr`: expression for β_x·δy binary-continuous products
- `x_disc::NMDTDiscretization`: discretization for x
- `y_disc::NMDTDiscretization`: discretization for y
- `meta::String`: identifier encoding the original variable type being approximated
- `lambda::Float64`: convex combination weight for the two NMDT estimates (default: `DNMDT_LAMBDA` = 0.5)
- `result_type`: expression type to store results in (default: `NMDTResultExpression`)
"""
function _assemble_dnmdt!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    bx_yh_expr,
    by_dx_expr,
    by_xh_expr,
    bx_dy_expr,
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String;
    lambda::Float64 = DNMDT_LAMBDA,
    result_type::Type = NMDTResultExpression,
    tighten::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        result_type,
        C,
        names,
        time_steps;
        meta,
    )

    dz = _residual_product!(
        container, C, names, time_steps,
        x_disc, y_disc.delta_var, 2.0^(-depth),
        depth, meta; tighten,
    )
    z1_expr = _assemble_product!(
        container, C, names, time_steps,
        [bx_yh_expr, by_dx_expr], dz,
        x_disc, y_disc, x_min, x_max, y_min, y_max,
        meta * "_nmdt1",
    )
    z2_expr = _assemble_product!(
        container, C, names, time_steps,
        [by_xh_expr, bx_dy_expr], dz,
        y_disc, x_disc, y_min, y_max, x_min, x_max,
        meta * "_nmdt2",
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(result, z1_expr[name, t], lambda)
        add_proportional_to_jump_expression!(result, z2_expr[name, t], 1.0 - lambda)
    end

    return result_expr
end
