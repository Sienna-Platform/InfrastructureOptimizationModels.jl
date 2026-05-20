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

"""
    build_quadratic_approx(config::SawtoothQuadConfig, model, x, x_min, x_max)

Scalar form: PWL approximation of x² for a single JuMP scalar `x` with
bounds `[x_min, x_max]`, using `depth` binary variables. If
`config.epigraph_depth > 0`, also builds an epigraph Q^{L1} lower bound and
tightens with z ≤ sawtooth_upper and z ≥ epigraph.

Returns a NamedTuple with `(approximation, g_var, alpha_var, link_constraint,
mip_constraints, tightening)` where `tightening` is `nothing` or
`(; z_var, constraints :: 1D over 1:2, epigraph)`.
"""
function build_quadratic_approx(
    config::SawtoothQuadConfig,
    model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
)
    @assert config.depth >= 1
    @assert x_max > x_min

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
    mip_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(undef, 1:depth, 1:4)
    @views mip_cons.data[:, 1] .= mip_a
    @views mip_cons.data[:, 2] .= mip_b
    @views mip_cons.data[:, 3] .= mip_c
    @views mip_cons.data[:, 4] .= mip_d

    x_sq_approx = JuMP.@expression(
        model,
        scale_back_g_basis(x_min, delta, g_var, 1:depth),
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
        tight_cons = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(undef, 1:2)
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

"""
    add_quadratic_approx!(config::SawtoothQuadConfig, container, ::Type{C}, x_var, x_bounds, meta)

Allocate sawtooth containers (g, α, link, mip, approximation) plus, when
`config.epigraph_depth > 0`, the tightened-z + 2-constraint containers AND
the full set of epigraph containers under `meta * "_lb"`. Loop `(name, t)`.
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
    @assert depth >= 1
    @assert length(name_axis) == length(x_bounds)
    for b in x_bounds
        @assert b.max > b.min
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
    local st_z_target, st_tight_target, epi_targets
    local epi_depth::Int
    if tighten
        st_z_target = add_variable_container!(
            container, SawtoothTightenedVariable, C, name_axis, time_axis; meta,
        )
        st_tight_target = add_constraints_container!(
            container, SawtoothTightenedConstraint, C, name_axis, 1:2, time_axis; meta,
        )
        epi_depth = config.epigraph_depth
        epi_targets = _alloc_epigraph_targets!(
            container, C, name_axis, time_axis, epi_depth, meta * "_lb",
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
                _write_epigraph_cell!(epi_targets, name, t, tt.epigraph, epi_depth)
            end
        end
    end
    return approx_target
end
