"Axes and axis data of one model. `dataset_axes!` adds component axes to it."
struct AxisRegistry
    axes::Dict{String, IS.AbstractAxis}
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

"Return `true` if a label contains the name delimiter."
is_encoded(labels::Vector{String}) = any(l -> occursin(PSI_NAME_DELIMITER, l), labels)

function _id_matrix(ids::Vector{Int64})
    if isempty(ids)
        return Matrix{Int64}(undef, 0, 0)
    end
    return reshape(ids, 1, :)
end

function _same_labels(reg::AxisRegistry, key::String, labels::Matrix{String})
    return reg.data[(reg.model, key)].labels == labels
end

"Add a component axis for `labels` to `reg`. Return the axis key."
function register_component_axis!(
    reg::AxisRegistry,
    sys,
    ::Type{U},
    labels::Vector{String},
    dataset_name::String,
) where {U}
    ids = Int64[]
    names = labels
    if !is_encoded(labels)
        (resolved_ids, names) = resolve_component_axis(sys, U, labels)
        ids = Vector{Int64}(resolved_ids)
    end
    has_ids = !isempty(ids)
    label_matrix = reshape(names, 1, :)
    for subset in ("", _lower_snake(dataset_name))
        columns = [IS.AxisColumn(string(nameof(U)), has_ids)]
        key =
            IS.axis_key(IS.ComponentAxis("component", columns, length(labels), "", subset))
        if !haskey(reg.axes, key)
            reg.axes[key] = IS.ComponentAxis(
                "component", columns, length(labels), "/$(reg.model)/axes/$key", subset,
            )
            reg.data[(reg.model, key)] =
                IS.AxisData(_id_matrix(ids), label_matrix, Dates.DateTime[])
            return key
        end
        if _same_labels(reg, key, label_matrix)
            return key
        end
    end
    throw(
        ArgumentError(
            "Axis for $(nameof(U)) already exists with different labels. Dataset: $dataset_name.",
        ),
    )
end

"Add a label axis for `labels` to `reg` under `key`. Return `key`."
function register_label_axis!(
    reg::AxisRegistry,
    labels::Vector{String},
    key::String,
    dataset_name::String,
)
    axis = IS.LabelAxis("label", labels)
    if !haskey(reg.axes, key)
        reg.axes[key] = axis
        return key
    end
    if reg.axes[key] == axis
        return key
    end
    throw(
        ArgumentError(
            "Label axis $key already exists with different labels. Dataset: $dataset_name.",
        ),
    )
end

function _label_axis_key(dataset_name::String, i::Int, count::Int)
    if isone(count)
        return _lower_snake(dataset_name)
    end
    return _lower_snake(dataset_name) * "_" * string(i)
end

"""
Add one axis per label vector of `key` to `reg`. The first label vectors, one per type of the
key component, are component axes. Every other label vector is a label axis. Return the axis
keys.
"""
function dataset_axes!(
    reg::AxisRegistry,
    sys,
    key::OptimizationContainerKey{T, U},
    labels::Vector{Vector{String}},
    dataset_name::String,
) where {T, U}
    types = _axis_types(U)
    axis_keys = String[]
    for (type, column) in zip(types, labels)
        push!(axis_keys, register_component_axis!(reg, sys, type, column, dataset_name))
    end
    extra = labels[(length(types) + 1):end]
    for (j, column) in enumerate(extra)
        i = length(types) + j
        label_key = _label_axis_key(dataset_name, i, length(extra))
        push!(axis_keys, register_label_axis!(reg, column, label_key, dataset_name))
    end
    return axis_keys
end
