mutable struct EmulationModel{M <: AbstractOptimizationProblem} <: AbstractOptimizationModel
    name::Symbol
    template::AbstractProblemTemplate
    sys::IS.InfrastructureSystemsContainer
    internal::ModelInternal
    simulation_info::SimulationInfo
    store::EmulationModelStore # might be extended to other stores for simulation
    ext::Dict{String, Any}
end

get_problem_type(::EmulationModel{M}) where {M <: AbstractOptimizationProblem} = M

function validate_template(::EmulationModel{M}) where {M <: AbstractOptimizationProblem}
    error("validate_template is not implemented for EmulationModel{$M}")
end

function get_current_time(model::EmulationModel)
    execution_count = get_execution_count(model)
    initial_time = get_initial_time(model)
    resolution = get_resolution(model)
    return initial_time + resolution * execution_count
end

function init_model_store_params!(model::EmulationModel)
    num_executions = get_executions(model)
    system = get_system(model)
    settings = get_settings(model)
    horizon = interval = resolution = get_resolution(settings)
    base_power = get_base_power(system)
    sys_uuid = get_system_uuid(system)
    set_store_params!(
        get_internal(model),
        ModelStoreParams(
            num_executions,
            horizon,
            interval,
            resolution,
            base_power,
            sys_uuid,
            get_metadata(get_optimization_container(model)),
        ),
    )
    return
end
