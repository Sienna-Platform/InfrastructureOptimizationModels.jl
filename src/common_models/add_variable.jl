@doc raw"""
Adds a variable to the optimization model and to the affine expressions contained
in the optimization_container model according to the specified sign. Based on the inputs, the variable can
be specified as binary.

# Bounds

``` lb_value_function <= varstart[name, t] <= ub_value_function ```

If binary = true:

``` varstart[name, t] in {0,1} ```

# LaTeX

``  lb \ge x^{device}_t \le ub \forall t ``

``  x^{device}_t \in {0,1} \forall t iff \text{binary = true}``

# Arguments
* container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
* devices : Vector or Iterator with the devices
* var_key::VariableKey : Base Name for the variable
* binary::Bool : Select if the variable is binary
* expression_name::Symbol : Expression_name name stored in container.expressions to add the variable
* sign::Float64 : sign of the addition of the variable to the expression_name. Default Value is 1.0

# Accepted Keyword Arguments
* ub_value : Provides the function over device to obtain the value for a upper_bound
* lb_value : Provides the function over device to obtain the value for a lower_bound. If the variable is meant to be positive define lb = x -> 0.0
* initial_value : Provides the function over device to obtain the warm start value

"""
function add_variables!(
    container::OptimizationContainer,
    ::Type{T},
    devices::U,
    ::Type{F},
) where {
    T <: VariableType,
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
    F,
} where {D <: IS.InfrastructureSystemsComponent}
    @assert !isempty(devices)
    time_steps = get_time_steps(container)
    settings = get_settings(container)
    binary = get_variable_binary(T, D, F)

    variable = add_variable_container!(
        container,
        T,
        D,
        get_name.(devices),
        time_steps,
    )

    for t in time_steps, d in devices
        name = get_name(d)
        variable[name, t] = JuMP.@variable(
            get_jump_model(container),
            base_name = "$(T)_$(D)_{$(name), $(t)}",
            binary = binary
        )
        ub = get_variable_upper_bound(T, d, F)
        ub !== nothing && JuMP.set_upper_bound(variable[name, t], ub)

        lb = get_variable_lower_bound(T, d, F)
        lb !== nothing && JuMP.set_lower_bound(variable[name, t], lb)

        if get_warm_start(settings)
            init = get_variable_warm_start_value(T, d, F)
            init !== nothing && JuMP.set_start_value(variable[name, t], init)
        end
    end

    return
end

"""
Add variables to the OptimizationContainer for a service.
"""
function add_service_variables!(
    container::OptimizationContainer,
    ::Type{T},
    service::U,
    contributing_devices::V,
    ::Type{F},
) where {
    T <: VariableType,
    U <: IS.InfrastructureSystemsComponent,
    V <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
    F <: AbstractServiceFormulation,
} where {D <: IS.InfrastructureSystemsComponent}
    @assert !isempty(contributing_devices)
    time_steps = get_time_steps(container)

    binary = get_variable_binary(T, U, F)

    variable = add_variable_container!(
        container,
        T,
        U,
        IS.get_name(service),
        [IS.get_name(d) for d in contributing_devices],
        time_steps,
    )

    for t in time_steps, d in contributing_devices
        name = IS.get_name(d)
        variable[name, t] = JuMP.@variable(
            get_jump_model(container),
            base_name = "$(T)_$(U)_$(IS.get_name(service))_{$(name), $(t)}",
            binary = binary
        )

        ub = get_variable_upper_bound(T, service, d, F)
        ub !== nothing && JuMP.set_upper_bound(variable[name, t], ub)

        lb = get_variable_lower_bound(T, service, d, F)
        lb !== nothing && !binary && JuMP.set_lower_bound(variable[name, t], lb)

        init = get_variable_warm_start_value(T, d, F)
        init !== nothing && JuMP.set_start_value(variable[name, t], init)
    end

    return
end
