"Axes and axis data of one model. `dataset_axes!` adds component axes to it."
struct AxisRegistry
    axes::Dict{String, InfrastructureOutputSchemas.Axis}
    data::Dict{Tuple{String, String}, IS.AxisData}
    model::String
end

"Return the column labels of `array` as one string vector per component axis."
function column_labels(key::OptimizationContainerKey, array::DenseAxisArray)
    return Vector{String}[collect(c) for c in get_column_names_from_axis_array(key, array)]
end

"Return `true` if the only column label is the encoded name of `key`."
function is_componentless(key::OptimizationContainerKey, labels::Vector{Vector{String}})
    return labels == collect(get_column_names_from_key(key))
end

"""
Return `true` if every label contains the name delimiter. A mix is plain names: a component
name can contain the delimiter.
"""
function is_encoded(labels::Vector{String})
    return !isempty(labels) && all(l -> occursin(PSI_NAME_DELIMITER, l), labels)
end

function _check_resolved(axis::String, ids, names, labels::Vector{String})
    if length(names) != length(labels) || (!isempty(ids) && length(ids) != length(labels))
        throw(
            ArgumentError(
                "resolve_component_axis for $axis returned $(length(ids)) ids and " *
                "$(length(names)) names for $(length(labels)) labels.",
            ),
        )
    end
    return
end

function _id_matrix(ids::Vector{Int64})
    if isempty(ids)
        return Matrix{Int64}(undef, 0, 0)
    end
    return reshape(ids, 1, :)
end

function _same_axis_data(
    reg::AxisRegistry,
    name::String,
    ids::Matrix{Int64},
    labels::Matrix{String},
)
    data = reg.data[(reg.model, name)]
    return data.ids == ids && data.labels == labels
end

function _component_axis(
    columns::Vector{InfrastructureOutputSchemas.ComponentAxisColumnsItem},
    n::Int,
    group::String,
    subset::String,
)
    if isempty(subset)
        return InfrastructureOutputSchemas.ComponentAxis(;
            kind = "component", columns = columns, length = n, group = group,
        )
    end
    return InfrastructureOutputSchemas.ComponentAxis(;
        kind = "component", columns = columns, length = n, group = group, subset = subset,
    )
end

"""
Add a component axis for `labels` to `reg`. Return the axis name. An axis with the same ids
and labels is used again. `position` is the position of the component type in the key: a
second component type of one dataset has its own fallback name.
"""
function register_component_axis!(
    reg::AxisRegistry,
    sys,
    ::Type{U},
    labels::Vector{String},
    dataset_name::String,
    position::Int = 1,
) where {U}
    ids = Int64[]
    names = labels
    if !is_encoded(labels)
        (resolved_ids, names) = resolve_component_axis(sys, U, labels)
        _check_resolved(string(nameof(U)), resolved_ids, names, labels)
        ids = Vector{Int64}(resolved_ids)
    end
    has_ids = !isempty(ids)
    id_matrix = _id_matrix(ids)
    label_matrix = reshape(names, 1, :)
    suffix = _lower_snake(dataset_name)
    subsets = ["", suffix]
    if position > 1
        push!(subsets, string(suffix, "_", position))
    end
    for subset in subsets
        columns = [
            InfrastructureOutputSchemas.ComponentAxisColumnsItem(;
                component_type = string(nameof(U)), has_ids = has_ids,
            ),
        ]
        name = IS.axis_name(_component_axis(columns, length(labels), "", subset))
        if !haskey(reg.axes, name)
            reg.axes[name] = InfrastructureOutputSchemas.Axis(
                _component_axis(
                    columns,
                    length(labels),
                    "/$(reg.model)/axes/$name",
                    subset,
                ),
            )
            reg.data[(reg.model, name)] =
                IS.AxisData(id_matrix, label_matrix, Dates.DateTime[])
            return name
        end
        if _same_axis_data(reg, name, id_matrix, label_matrix)
            return name
        end
    end
    throw(
        ArgumentError(
            "Axis for $(nameof(U)) already exists with different labels or ids. Dataset: $dataset_name.",
        ),
    )
end

_has_labels(a::InfrastructureOutputSchemas.Axis, labels::Vector{String}) =
    _has_labels(a.value, labels)
_has_labels(a::InfrastructureOutputSchemas.LabelAxis, labels::Vector{String}) =
    a.labels == labels
_has_labels(::Any, ::Vector{String}) = false

"Add a label axis for `labels` to `reg` under `name`. Return `name`."
function register_label_axis!(
    reg::AxisRegistry,
    labels::Vector{String},
    name::String,
    dataset_name::String,
)
    if !haskey(reg.axes, name)
        reg.axes[name] = InfrastructureOutputSchemas.Axis(
            InfrastructureOutputSchemas.LabelAxis(;
                kind = "label",
                name = name,
                labels = labels,
            ),
        )
        return name
    end
    if _has_labels(reg.axes[name], labels)
        return name
    end
    throw(
        ArgumentError(
            "Label axis $name already exists with different labels. Dataset: $dataset_name.",
        ),
    )
end

function _label_axis_name(dataset_name::String, i::Int, count::Int)
    if isone(count)
        return _lower_snake(dataset_name)
    end
    return _lower_snake(dataset_name) * "_" * string(i)
end

"""
Add one axis per label vector of `key` to `reg`. The first label vectors, one per type of the
key component, are component axes. Every other label vector is a label axis. Return the axis
names.
"""
function dataset_axes!(
    reg::AxisRegistry,
    sys,
    key::OptimizationContainerKey{T, U},
    labels::Vector{Vector{String}},
    dataset_name::String,
) where {T, U}
    types = _axis_types(U)
    axis_names = String[]
    for (position, (type, column)) in enumerate(zip(types, labels))
        push!(
            axis_names,
            register_component_axis!(reg, sys, type, column, dataset_name, position),
        )
    end
    extra = labels[(length(types) + 1):end]
    for (j, column) in enumerate(extra)
        i = length(types) + j
        label_axis_name = _label_axis_name(dataset_name, i, length(extra))
        push!(axis_names, register_label_axis!(reg, column, label_axis_name, dataset_name))
    end
    return axis_names
end
