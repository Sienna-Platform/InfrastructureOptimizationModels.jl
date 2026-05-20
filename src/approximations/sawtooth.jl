# Sawtooth MIP approximation of x² for use in constraints.
# Uses recursive tooth function compositions with O(log(1/ε)) binary variables.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024).

"Binary variables (α₁, …, α_L) for sawtooth quadratic approximation."
struct SawtoothBinaryVariable <: VariableType end

"Variable result in tightened version."
struct SawtoothTightenedVariable <: VariableType end

"Constrains g_j based on g_{j-1}."
struct SawtoothMIPConstraint <: ConstraintType end

"Bounds tightened sawtooth variable."
struct SawtoothTightenedConstraint <: ConstraintType end

"""
Config for sawtooth MIP quadratic approximation.

# Fields
- `depth::Int`: recursion depth L; uses L binary variables for 2^L + 1 breakpoints.
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound;
  0 to disable (default 0).
"""
struct SawtoothQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
function SawtoothQuadConfig(depth::Int)
    return SawtoothQuadConfig(depth, 0)
end

# --- Scalar build (pure JuMP, primary API) ---

"""
    build_quadratic_approx(config::SawtoothQuadConfig, model, x, x_min, x_max)

Scalar form: PWL approximation of x² for a single JuMP scalar `x` with
bounds `[x_min, x_max]`, using `depth` binary variables. If
`config.epigraph_depth > 0`, also builds an epigraph Q^{L1} lower bound and
tightens the approximation via z ≤ sawtooth_upper and z ≥ epigraph.

Returns a NamedTuple:
- `approximation`   :: scalar (AffExpr; either `x_sq_approx` or `1.0·z` if tightened)
- `g_var`           :: DenseAxisArray{VariableRef, 1} over `0:depth`
- `alpha_var`       :: DenseAxisArray{VariableRef, 1} over `1:depth`
- `link_constraint` :: scalar constraint linking g₀ to (x − x_min)/δ
- `mip_constraints` :: DenseAxisArray{Constraint, 2} over `(1:depth, 1:4)`
- `tightening`      :: `nothing`, or a NamedTuple
                       `(; z_var, constraints :: 1D over 1:2, epigraph)` where
                       `epigraph` is the full NamedTuple returned by the
                       scalar epigraph build.
"""
function build_quadratic_approx(
    config::SawtoothQuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    IS.@assert_op config.depth >= 1
    IS.@assert_op x_max > x_min

    depth = config.depth
    delta = x_max - x_min

    g_var = JuMP.@variable(
        model, [j = 0:depth],
        lower_bound = 0.0, upper_bound = 1.0,
        base_name = "SawtoothAux",
    )
    alpha_var = JuMP.@variable(
        model, [j = 1:depth],
        binary = true,
        base_name = "SawtoothBin",
    )

    link_con = JuMP.@constraint(model, g_var[0] == (x - x_min) / delta)

    # S^L constraints: 4 inequalities per level.
    mip_a = JuMP.@constraint(
        model, [j = 1:depth], g_var[j] <= 2.0 * g_var[j - 1],
    )
    mip_b = JuMP.@constraint(
        model, [j = 1:depth], g_var[j] <= 2.0 * (1.0 - g_var[j - 1]),
    )
    mip_c = JuMP.@constraint(
        model, [j = 1:depth], g_var[j] >= 2.0 * (g_var[j - 1] - alpha_var[j]),
    )
    mip_d = JuMP.@constraint(
        model, [j = 1:depth], g_var[j] >= 2.0 * (alpha_var[j] - g_var[j - 1]),
    )
    mip_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
        undef, 1:depth, 1:4,
    )
    @views mip_cons.data[:, 1] .= mip_a.data
    @views mip_cons.data[:, 2] .= mip_b.data
    @views mip_cons.data[:, 3] .= mip_c.data
    @views mip_cons.data[:, 4] .= mip_d.data

    x_sq_approx = JuMP.@expression(
        model,
        scale_back_g_basis_scalar(x_min, delta, g_var, 1:depth),
    )

    if config.epigraph_depth > 0
        epi = build_quadratic_approx(
            EpigraphQuadConfig(config.epigraph_depth), model, x, x_min, x_max,
        )
        z_min = (x_min <= 0.0 <= x_max) ? 0.0 : min(x_min^2, x_max^2)
        z_max = max(x_min^2, x_max^2)
        z_var = JuMP.@variable(
            model, lower_bound = z_min, upper_bound = z_max,
            base_name = "TightenedSawtooth",
        )
        tight_a = JuMP.@constraint(model, z_var <= x_sq_approx)
        tight_b = JuMP.@constraint(model, z_var >= epi.approximation)
        tight_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
            undef, 1:2,
        )
        tight_cons[1] = tight_a
        tight_cons[2] = tight_b
        approximation = JuMP.@expression(model, 1.0 * z_var)
        tightening = (; z_var, constraints = tight_cons, epigraph = epi)
        return (;
            approximation,
            g_var,
            alpha_var,
            link_constraint = link_con,
            mip_constraints = mip_cons,
            tightening,
        )
    end

    return (;
        approximation = x_sq_approx,
        g_var,
        alpha_var,
        link_constraint = link_con,
        mip_constraints = mip_cons,
        tightening = nothing,
    )
end

# --- IOM adapter (allocate, loop, write) ---

"""
    add_quadratic_approx!(config::SawtoothQuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate sawtooth containers (g, α, link, mip, approximation) plus, when
`config.epigraph_depth > 0`, the tightened-z + 2-constraint containers AND
the full set of epigraph containers under `meta * "_lb"`. Then loop
`(name, t)` calling the scalar build per cell and writing all the refs
into their slots.

Returns the registered `QuadraticExpression` container.
"""
function add_quadratic_approx!(
    config::SawtoothQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    x_var,
    x_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(x_var, 1)
    time_axis = axes(x_var, 2)
    depth = config.depth
    IS.@assert_op depth >= 1
    IS.@assert_op length(name_axis) == length(x_bounds)
    for b in x_bounds
        IS.@assert_op b.max > b.min
    end

    model = get_jump_model(container)

    g_target = add_variable_container!(
        container, SawtoothAuxVariable, C, name_axis, 0:depth, time_axis; meta,
    )
    alpha_target = add_variable_container!(
        container, SawtoothBinaryVariable, C, name_axis, 1:depth, time_axis; meta,
    )
    link_target = add_constraints_container!(
        container, SawtoothLinkingConstraint, C, name_axis, time_axis; meta,
    )
    mip_target = add_constraints_container!(
        container, SawtoothMIPConstraint, C, name_axis, 1:depth, 1:4, time_axis; meta,
    )
    approx_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
    )

    tighten = config.epigraph_depth > 0
    local st_z_target, st_tight_target
    local epi_z_target, epi_g_target, epi_link_target, epi_fL_target,
        epi_approx_target, epi_lp_target, epi_tangent_target
    local epi_depth::Int
    if tighten
        st_z_target = add_variable_container!(
            container, SawtoothTightenedVariable, C, name_axis, time_axis; meta,
        )
        st_tight_target = add_constraints_container!(
            container, SawtoothTightenedConstraint, C, name_axis, 1:2, time_axis; meta,
        )
        epi_depth = config.epigraph_depth
        epi_meta = meta * "_lb"
        epi_z_target = add_variable_container!(
            container, EpigraphVariable, C, name_axis, time_axis; meta = epi_meta,
        )
        epi_g_target = add_variable_container!(
            container, SawtoothAuxVariable, C, name_axis, 0:epi_depth, time_axis;
            meta = epi_meta,
        )
        epi_link_target = add_constraints_container!(
            container, SawtoothLinkingConstraint, C, name_axis, time_axis;
            meta = epi_meta,
        )
        epi_fL_target = add_expression_container!(
            container, EpigraphTangentExpression, C, name_axis, time_axis;
            meta = epi_meta,
        )
        epi_approx_target = add_expression_container!(
            container, EpigraphExpression, C, name_axis, time_axis; meta = epi_meta,
        )
        epi_lp_target = add_constraints_container!(
            container, SawtoothLPConstraint, C, name_axis, 1:epi_depth, 1:2, time_axis;
            meta = epi_meta,
        )
        epi_tangent_target = add_constraints_container!(
            container, EpigraphTangentConstraint, C, name_axis, 1:(epi_depth + 2),
            time_axis; meta = epi_meta,
        )
    end

    for (i, name) in enumerate(name_axis)
        xmn, xmx = x_bounds[i].min, x_bounds[i].max
        for t in time_axis
            r = build_quadratic_approx(config, model, x_var[name, t], xmn, xmx)
            for j in 0:depth
                g_target[name, j, t] = r.g_var[j]
            end
            for j in 1:depth
                alpha_target[name, j, t] = r.alpha_var[j]
            end
            link_target[name, t] = r.link_constraint
            for j in 1:depth, k in 1:4
                mip_target[name, j, k, t] = r.mip_constraints[j, k]
            end
            approx_target[name, t] = r.approximation

            if tighten
                tt = r.tightening
                st_z_target[name, t] = tt.z_var
                for k in 1:2
                    st_tight_target[name, k, t] = tt.constraints[k]
                end
                epi = tt.epigraph
                epi_z_target[name, t] = epi.z_var
                for j in 0:epi_depth
                    epi_g_target[name, j, t] = epi.g_var[j]
                end
                epi_link_target[name, t] = epi.link_constraint
                epi_fL_target[name, t] = epi.tangent_expression
                epi_approx_target[name, t] = epi.approximation
                for j in 1:epi_depth, k in 1:2
                    epi_lp_target[name, j, k, t] = epi.lp_constraints[j, k]
                end
                for j in 1:(epi_depth + 2)
                    epi_tangent_target[name, j, t] = epi.tangent_constraints[j]
                end
            end
        end
    end
    return approx_target
end

# --- Legacy result + tightening structs + vectorized build + register
# (kept for the generic add_quadratic_approx! wrapper in common.jl until
# callers migrate; removed in sweep) ---

"""
Tightening pieces of a sawtooth result when `config.epigraph_depth > 0`:
the substitute z variable, its bound constraints, and the epigraph result
that supplies the lower bound (legacy).
"""
struct SawtoothTightening{
    ZV <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 2},
    TC <: JuMP.Containers.DenseAxisArray,
    EPI <: EpigraphQuadResult,
}
    z_var::ZV
    constraints::TC
    epigraph::EPI
end

"""
Pure-JuMP result of legacy vectorized `build_quadratic_approx(::SawtoothQuadConfig, ...)`.
"""
struct SawtoothQuadResult{
    A <: JuMP.Containers.DenseAxisArray,
    G <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    AL <: JuMP.Containers.DenseAxisArray{JuMP.VariableRef, 3},
    LC <: JuMP.Containers.DenseAxisArray,
    MC <: JuMP.Containers.DenseAxisArray,
    T <: Union{Nothing, SawtoothTightening},
} <: QuadraticApproxResult
    approximation::A
    g_var::G
    alpha_var::AL
    link_constraints::LC
    mip_constraints::MC
    tightening::T
end

"""
    build_quadratic_approx(config::SawtoothQuadConfig, model, x, bounds)

Legacy vectorized form. PWL approximation of x² with sawtooth tooth
functions and L binary variables. If `config.epigraph_depth > 0`, also
builds an epigraph Q^{L1} lower bound and tightens the approximation:
z ≤ x² (sawtooth, upper) and z ≥ epigraph (lower).
"""
function build_quadratic_approx(
    config::SawtoothQuadConfig,
    model::JuMP.Model,
    x,
    bounds::Vector{MinMax},
)
    IS.@assert_op config.depth >= 1
    name_axis = axes(x, 1)
    time_axis = axes(x, 2)
    IS.@assert_op length(name_axis) == length(bounds)
    for b in bounds
        IS.@assert_op b.max > b.min
    end

    g_levels = 0:(config.depth)
    alpha_levels = 1:(config.depth)
    delta = JuMP.Containers.DenseAxisArray([b.max - b.min for b in bounds], name_axis)
    x_min_arr = JuMP.Containers.DenseAxisArray([b.min for b in bounds], name_axis)

    g_var = JuMP.@variable(
        model,
        [name = name_axis, j = g_levels, t = time_axis],
        lower_bound = 0.0,
        upper_bound = 1.0,
        base_name = "SawtoothAux",
    )
    alpha_var = JuMP.@variable(
        model,
        [name = name_axis, j = alpha_levels, t = time_axis],
        binary = true,
        base_name = "SawtoothBin",
    )

    link_cons = JuMP.@constraint(
        model,
        [name = name_axis, t = time_axis],
        g_var[name, 0, t] == (x[name, t] - x_min_arr[name]) / delta[name],
    )

    mip_a = JuMP.@constraint(
        model,
        [name = name_axis, j = alpha_levels, t = time_axis],
        g_var[name, j, t] <= 2.0 * g_var[name, j - 1, t],
    )
    mip_b = JuMP.@constraint(
        model,
        [name = name_axis, j = alpha_levels, t = time_axis],
        g_var[name, j, t] <= 2.0 * (1.0 - g_var[name, j - 1, t]),
    )
    mip_c = JuMP.@constraint(
        model,
        [name = name_axis, j = alpha_levels, t = time_axis],
        g_var[name, j, t] >= 2.0 * (g_var[name, j - 1, t] - alpha_var[name, j, t]),
    )
    mip_d = JuMP.@constraint(
        model,
        [name = name_axis, j = alpha_levels, t = time_axis],
        g_var[name, j, t] >= 2.0 * (alpha_var[name, j, t] - g_var[name, j - 1, t]),
    )
    mip_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
        undef, name_axis, alpha_levels, 1:4, time_axis,
    )
    @views mip_cons.data[:, :, 1, :] .= mip_a.data
    @views mip_cons.data[:, :, 2, :] .= mip_b.data
    @views mip_cons.data[:, :, 3, :] .= mip_c.data
    @views mip_cons.data[:, :, 4, :] .= mip_d.data

    x_sq_approx = JuMP.@expression(
        model,
        [name = name_axis, t = time_axis],
        scale_back_g_basis(
            x_min_arr[name], delta[name], g_var, name, t, alpha_levels,
        )
    )

    if config.epigraph_depth > 0
        epi_result = build_quadratic_approx(
            EpigraphQuadConfig(config.epigraph_depth), model, x, bounds,
        )
        z_min_arr = JuMP.Containers.DenseAxisArray(
            [(b.min <= 0.0 <= b.max) ? 0.0 : min(b.min^2, b.max^2) for b in bounds],
            name_axis,
        )
        z_max_arr = JuMP.Containers.DenseAxisArray(
            [max(b.min^2, b.max^2) for b in bounds],
            name_axis,
        )
        z_var = JuMP.@variable(
            model,
            [name = name_axis, t = time_axis],
            lower_bound = z_min_arr[name],
            upper_bound = z_max_arr[name],
            base_name = "TightenedSawtooth",
        )
        tight_a = JuMP.@constraint(
            model,
            [name = name_axis, t = time_axis],
            z_var[name, t] <= x_sq_approx[name, t],
        )
        tight_b = JuMP.@constraint(
            model,
            [name = name_axis, t = time_axis],
            z_var[name, t] >= epi_result.approximation[name, t],
        )
        tight_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
            undef, name_axis, 1:2, time_axis,
        )
        @views tight_cons.data[:, 1, :] .= tight_a.data
        @views tight_cons.data[:, 2, :] .= tight_b.data
        approximation = JuMP.@expression(
            model,
            [name = name_axis, t = time_axis],
            1.0 * z_var[name, t]
        )
        tightening = SawtoothTightening(z_var, tight_cons, epi_result)
        return SawtoothQuadResult(
            approximation, g_var, alpha_var, link_cons, mip_cons, tightening,
        )
    end

    return SawtoothQuadResult(
        x_sq_approx, g_var, alpha_var, link_cons, mip_cons, nothing,
    )
end

function register_in_container!(
    container::OptimizationContainer,
    ::Type{C},
    result::SawtoothQuadResult,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(result.approximation, 1)
    time_axis = axes(result.approximation, 2)
    g_levels = axes(result.g_var, 2)
    alpha_levels = axes(result.alpha_var, 2)

    g_target = add_variable_container!(
        container, SawtoothAuxVariable, C, name_axis, g_levels, time_axis; meta,
    )
    g_target.data .= result.g_var.data

    alpha_target = add_variable_container!(
        container, SawtoothBinaryVariable, C, name_axis, alpha_levels, time_axis; meta,
    )
    alpha_target.data .= result.alpha_var.data

    link_target = add_constraints_container!(
        container, SawtoothLinkingConstraint, C, name_axis, time_axis; meta,
    )
    link_target.data .= result.link_constraints.data

    mip_target = add_constraints_container!(
        container, SawtoothMIPConstraint, C, name_axis, alpha_levels, 1:4, time_axis;
        meta,
    )
    mip_target.data .= result.mip_constraints.data

    result_target = add_expression_container!(
        container, QuadraticExpression, C, name_axis, time_axis; meta,
    )
    result_target.data .= result.approximation.data

    _register_sawtooth_tightening!(container, C, result.tightening, meta)
    return
end

function _register_sawtooth_tightening!(
    container::OptimizationContainer,
    ::Type{C},
    tight::SawtoothTightening,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    name_axis = axes(tight.z_var, 1)
    time_axis = axes(tight.z_var, 2)
    z_target = add_variable_container!(
        container, SawtoothTightenedVariable, C, name_axis, time_axis; meta,
    )
    z_target.data .= tight.z_var.data
    tight_target = add_constraints_container!(
        container, SawtoothTightenedConstraint, C, name_axis, 1:2, time_axis; meta,
    )
    tight_target.data .= tight.constraints.data
    register_in_container!(container, C, tight.epigraph, meta * "_lb")
    return
end

# No-op when tightening is disabled (config.epigraph_depth = 0).
_register_sawtooth_tightening!(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Nothing,
    ::String,
) = nothing
