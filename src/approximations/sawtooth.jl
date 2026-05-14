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
Tightening pieces of a sawtooth result when `config.epigraph_depth > 0`:
the substitute z variable, its bound constraints, and the epigraph result
that supplies the lower bound.
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
Pure-JuMP result of `build_quadratic_approx(::SawtoothQuadConfig, ...)`.
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

PWL approximation of x² with sawtooth tooth functions and L binary variables.
If `config.epigraph_depth > 0`, also builds an epigraph Q^{L1} lower bound and
tightens the approximation: z ≤ x² (sawtooth, upper) and z ≥ epigraph (lower).
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

    # S^L constraints for j = 1..L: 4 inequalities per level. Stack four
    # `(name, j, t)` families into a `(name, j, k, t)` container.
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

    # x² ≈ x_min² + (2·x_min·δ + δ²)·g₀ − Σ_{j ∈ alpha_levels} δ²·2^{−2j}·g_j
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
        # tight_a is `z <= sawtooth approx` (LessThan), tight_b is `z >= epigraph`
        # (GreaterThan) — use the abstract ConstraintRef to hold both kinds.
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
