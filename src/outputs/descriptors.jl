"""
Return the ids and the component names of the `labels` of type `U` in `sys`, in the order of
`labels`. An empty id vector gives a label-only axis. A system type extends this.
"""
function resolve_component_axis(sys, ::Type{U}, labels::Vector{String}) where {U}
    throw(
        ArgumentError(
            "resolve_component_axis is not implemented for system type $(typeof(sys)).",
        ),
    )
end

_param_name(p::Type) = string(nameof(p))
_param_name(p) = string(p)

"Return the names of the type parameters of `T`. A type parameter that is not a type gives its `string`."
function _type_params(::Type{T}) where {T}
    return String[_param_name(p) for p in Base.unwrap_unionall(T).parameters]
end

_component_parts(::Type{U}) where {U} = _type_params(U)

function _component_parts(::Type{ComponentPairKey{A, B}}) where {A, B}
    return vcat(_type_params(A), [string(nameof(B))], _type_params(B))
end

"""
Return the dataset meta of a key with the types `T` and `U` and the meta `meta`. The meta
joins with `_` the names of the type parameters of `T`, the names of the type parameters of
`U` (a pair key adds the second type name and its parameters), and `meta`. Empty parts are
skipped.
"""
function _dataset_meta(::Type{T}, ::Type{U}, meta::String) where {T, U}
    parts = vcat(_type_params(T), _component_parts(U), [meta])
    return join(filter(!isempty, parts), "_")
end

"Return the dataset meta of `key`. The type parameters of the key types keep the datasets apart."
function get_meta(key::OptimizationContainerKey{T, U}) where {T, U}
    return _dataset_meta(T, U, key.meta)
end

"Return the tuple of component types that form the axes of component type `U`."
_axis_types(::Type{U}) where {U} = (U,)
_axis_types(::Type{ComponentPairKey{A, B}}) where {A, B} = (A, B)

"Return the `IS.OutputDataset` of `key`. The first axis type of `U` gives the component type."
function dataset_descriptor(
    key::OptimizationContainerKey{T, U};
    axes::Vector{<:IS.OutputAxis},
) where {T, U}
    return IS.dataset_descriptor(T, first(_axis_types(U)), get_meta(key), axes)
end

"Return the `IS.OutputDataset` of `key` for a dataset that has no component."
function dataset_descriptor_componentless(
    key::OptimizationContainerKey{T, U};
    axes::Vector{<:IS.OutputAxis},
) where {T, U}
    return IS.dataset_descriptor(T, get_meta(key), axes)
end

"Return the model type name that the outputs metadata stores for a store."
model_type_name(::DecisionModelStore) = "decision"
model_type_name(::EmulationModelStore) = "emulation"

"Return true when a reader can stitch the datasets of many executions into one time axis."
supports_realized_values_post_calculation(::EmulationModelStore, ::ModelStoreParams) = false

function supports_realized_values_post_calculation(
    ::DecisionModelStore,
    params::ModelStoreParams,
)
    return get_num_executions(params) > 1
end

"Return the model axes of a decision model store."
function model_axes(
    ::DecisionModelStore,
    params::ModelStoreParams,
    initial_time::Dates.DateTime,
)
    return IS.OutputAxis[
        IS.ExecutionAxis(initial_time, get_interval(params), get_num_executions(params)),
        IS.HorizonAxis(get_resolution(params), get_horizon_count(params)),
    ]
end

"Return the model axes of an emulation model store."
function model_axes(
    ::EmulationModelStore,
    params::ModelStoreParams,
    initial_time::Dates.DateTime,
)
    return IS.OutputAxis[
        IS.TimeAxis(initial_time, get_resolution(params), get_num_executions(params)),
    ]
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
