# Incremental piecewise linear (PWL) formulation utility.
#
# This is not an approximation method in the `build_quadratic_approx`/
# `build_bilinear_approx` sense — it's a container-coupled utility used by
# downstream packages (e.g. POM HVDC models) to build PWL variables and
# constraints for arbitrary nonlinear functions. Kept here because the
# math is in the same family as the other PWL approximations.

"""
    add_sparse_pwl_interpolation_variables!(container, ::Type{T}, devices, model, num_segments)

Add piecewise linear interpolation variables to an optimization container.

For continuous interpolation variables (`T <: InterpolationVariableType`),
creates `num_segments` variables per (device, t). For binary variables
(`T <: BinaryInterpolationVariableType`), creates `num_segments - 1`
variables (controlling transitions between segments).

# Arguments
- `container::OptimizationContainer`: target container.
- `::Type{T}`: variable type (interpolation or binary interpolation).
- `devices`: iterable of components.
- `model::DeviceModel{U, V}`: device model providing variable bounds.
- `num_segments`: number of PWL segments (default `DEFAULT_INTERPOLATION_LENGTH`).
"""
function add_sparse_pwl_interpolation_variables!(
    container::OptimizationContainer,
    ::Type{T},
    devices,
    model::DeviceModel{U, V},
    num_segments = DEFAULT_INTERPOLATION_LENGTH,
) where {
    T <: Union{InterpolationVariableType, BinaryInterpolationVariableType},
    U <: IS.InfrastructureSystemsComponent,
    V <: AbstractDeviceFormulation,
}
    time_steps = get_time_steps(container)
    var_container = lazy_container_addition!(container, T, U)
    binary_flag = get_variable_binary(T, U, V)
    len_segs = binary_flag ? (num_segments - 1) : num_segments

    for d in devices
        name = get_name(d)
        for t in time_steps
            for i in 1:len_segs
                var_container[(name, i, t)] = JuMP.@variable(
                    get_jump_model(container),
                    base_name = "$(T)_$(name)_{pwl_$(i), $(t)}",
                    binary = binary_flag
                )
                ub = get_variable_upper_bound(T, d, V)
                ub !== nothing && JuMP.set_upper_bound(var_container[name, i, t], ub)
                lb = get_variable_lower_bound(T, d, V)
                lb !== nothing && JuMP.set_lower_bound(var_container[name, i, t], lb)
            end
        end
    end
    return
end

"""
    _add_generic_incremental_interpolation_constraint!(container, ::R, ::S, ::T, ::U, ::V, devices, dic_var_bkpts, dic_function_bkpts; meta)

Add incremental piecewise linear interpolation constraints relating the
original variable x (type `R`) to its piecewise approximation y = f(x)
(type `S`), using interpolation variables δ (type `T`) and binary variables
z (type `U`) under constraint type `V`.

The incremental method represents each segment as:
- `x = x₁ + Σᵢ δᵢ · (xᵢ₊₁ − xᵢ)` with δᵢ ∈ [0, 1]
- `y = y₁ + Σᵢ δᵢ · (yᵢ₊₁ − yᵢ)`

Binary variables z enforce the incremental ordering δᵢ₊₁ ≤ zᵢ ≤ δᵢ.

# Arguments
- `container::OptimizationContainer`: target container.
- `::Type{R}`, `::Type{S}`: original and approximated variable types.
- `::Type{T}`, `::Type{U}`: interpolation and binary interpolation types.
- `::Type{V}`: constraint type.
- `devices`: iterable of components.
- `dic_var_bkpts::Dict{String, Vector{Float64}}`: domain breakpoints.
- `dic_function_bkpts::Dict{String, Vector{Float64}}`: function-value breakpoints.
- `meta`: constraint-name prefix (default `CONTAINER_KEY_EMPTY_META`).
"""
function _add_generic_incremental_interpolation_constraint!(
    container::OptimizationContainer,
    ::Type{R},
    ::Type{S},
    ::Type{T},
    ::Type{U},
    ::Type{V},
    devices::IS.FlattenIteratorWrapper{W},
    dic_var_bkpts::Dict{String, Vector{Float64}},
    dic_function_bkpts::Dict{String, Vector{Float64}};
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    R <: VariableType,
    S <: VariableType,
    T <: VariableType,
    U <: VariableType,
    V <: ConstraintType,
    W <: IS.InfrastructureSystemsComponent,
}
    time_steps = get_time_steps(container)
    names = [get_name(d) for d in devices]
    JuMPmodel = get_jump_model(container)

    x_var = if R <: DCVoltage
        get_variable(container, R, component_for_hvdc_interpolation(nothing))
    else
        get_variable(container, R, W)
    end
    y_var = get_variable(container, S, W)
    δ_var = get_variable(container, T, W)
    z_var = get_variable(container, U, W)

    const_container_var = add_constraints_container!(
        container,
        V,
        W,
        names,
        time_steps;
        meta = "$(meta)pwl_variable",
    )
    const_container_function = add_constraints_container!(
        container,
        V,
        W,
        names,
        time_steps;
        meta = "$(meta)pwl_function",
    )

    for d in devices
        name = get_name(d)
        x_name = (R <: DCVoltage) ? get_name(get_dc_bus(d)) : name
        var_bkpts = dic_var_bkpts[name]
        function_bkpts = dic_function_bkpts[name]
        num_segments = length(var_bkpts) - 1

        for t in time_steps
            const_container_var[name, t] = JuMP.@constraint(
                JuMPmodel,
                x_var[x_name, t] ==
                var_bkpts[1] + sum(
                    δ_var[name, i, t] * (var_bkpts[i + 1] - var_bkpts[i]) for
                    i in 1:num_segments
                )
            )
            const_container_function[name, t] = JuMP.@constraint(
                JuMPmodel,
                y_var[name, t] ==
                function_bkpts[1] + sum(
                    δ_var[name, i, t] * (function_bkpts[i + 1] - function_bkpts[i]) for
                    i in 1:num_segments
                )
            )
            for i in 1:(num_segments - 1)
                JuMP.@constraint(JuMPmodel, z_var[name, i, t] >= δ_var[name, i + 1, t])
                JuMP.@constraint(JuMPmodel, z_var[name, i, t] <= δ_var[name, i, t])
            end
        end
    end
    return
end
