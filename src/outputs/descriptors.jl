"Return the labels of `sys` components of type `U` as integer ids. A system type extends this."
function component_ids(sys, ::Type{U}, labels::Vector{String}) where {U}
    throw(
        ArgumentError(
            "component_ids is not implemented for system type $(typeof(sys)).",
        ),
    )
end

"Return the meta string of `key`."
get_meta(key::OptimizationContainerKey) = key.meta

"Return the tuple of component types that form the axes of component type `U`."
_axis_types(::Type{U}) where {U} = (U,)
_axis_types(::Type{ComponentPairKey{A, B}}) where {A, B} = (A, B)

"Return the `IS.OutputDataset` of `key`. The first axis type of `U` gives the component type."
function dataset_descriptor(
    key::OptimizationContainerKey{T, U};
    model::String,
    axes::Vector{String},
    axis_lengths::Vector{Int},
) where {T, U}
    return IS.dataset_descriptor(
        T,
        first(_axis_types(U)),
        get_meta(key);
        model = model,
        axes = axes,
        axis_lengths = axis_lengths,
    )
end

"Return the `IS.OutputDataset` of `key` for a dataset that has no component."
function dataset_descriptor_componentless(
    key::OptimizationContainerKey{T, U};
    model::String,
    axes::Vector{String},
    axis_lengths::Vector{Int},
) where {T, U}
    return IS.dataset_descriptor(
        T,
        get_meta(key);
        model = model,
        axes = axes,
        axis_lengths = axis_lengths,
    )
end

"Return the model type name that the outputs metadata stores for a store."
model_type_name(::DecisionModelStore) = "decision"
model_type_name(::EmulationModelStore) = "emulation"

"Return the model axes and the axis data of a decision model store."
function model_axes(
    ::DecisionModelStore,
    params::ModelStoreParams,
    initial_time::Dates.DateTime,
    model::String,
)
    interval = get_interval(params)
    num_executions = get_num_executions(params)
    execution = IS.ExecutionAxis(
        "execution",
        initial_time,
        IS.iso_duration(interval),
        num_executions,
        "/$model/axes/execution",
    )
    horizon = IS.HorizonAxis(
        "horizon",
        IS.iso_duration(get_resolution(params)),
        get_horizon_count(params),
    )
    timestamps =
        collect(range(initial_time; step = interval, length = num_executions))
    axes = Dict{String, IS.AbstractAxis}("execution" => execution, "horizon" => horizon)
    data = Dict{Tuple{String, String}, IS.AxisData}(
        (model, "execution") => IS.AxisData(timestamps),
    )
    return (axes, data)
end

"Return the model axes and the axis data of an emulation model store."
function model_axes(
    ::EmulationModelStore,
    params::ModelStoreParams,
    initial_time::Dates.DateTime,
    model::String,
)
    resolution = get_resolution(params)
    num_executions = get_num_executions(params)
    time_axis = IS.TimeAxis(
        "time",
        initial_time,
        IS.iso_duration(resolution),
        num_executions,
        "/$model/axes/time",
    )
    timestamps =
        collect(range(initial_time; step = resolution, length = num_executions))
    axes = Dict{String, IS.AbstractAxis}("time" => time_axis)
    data = Dict{Tuple{String, String}, IS.AxisData}(
        (model, "time") => IS.AxisData(timestamps),
    )
    return (axes, data)
end

_add_optimizer!(::Dict{String, Any}, ::Nothing) = nothing
function _add_optimizer!(dict::Dict{String, Any}, optimizer)
    dict["optimizer"] = string(optimizer.optimizer_constructor)
    return nothing
end

"Return the settings of `settings` that the outputs metadata stores."
function settings_dict(settings::Settings)
    dict = Dict{String, Any}(
        "horizon" => IS.iso_duration(get_horizon(settings)),
        "interval" => IS.iso_duration(get_interval(settings)),
        "resolution" => IS.iso_duration(get_resolution(settings)),
    )
    _add_optimizer!(dict, get_optimizer(settings))
    return dict
end

"Return `name` in lower snake case. Each character outside `[a-z0-9]` becomes `_`."
function _lower_snake(name::String)
    result = replace(lowercase(name), r"[^a-z0-9]" => "_")
    if isempty(result) || isdigit(first(result)) || first(result) == '_'
        result = "k" * result
    end
    return result
end
