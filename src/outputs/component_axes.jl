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
    has_ids = !is_encoded(labels)
    label_matrix = reshape(labels, 1, :)
    for subset in ("", _lower_snake(dataset_name))
        columns = [IS.AxisColumn(string(nameof(U)), has_ids)]
        key =
            IS.axis_key(IS.ComponentAxis("component", columns, length(labels), "", subset))
        if !haskey(reg.axes, key)
            ids = Matrix{Int64}(undef, 0, 0)
            if has_ids
                ids = reshape(Vector{Int64}(component_ids(sys, U, labels)), 1, :)
            end
            reg.axes[key] = IS.ComponentAxis(
                "component", columns, length(labels), "/$(reg.model)/axes/$key", subset,
            )
            reg.data[(reg.model, key)] = IS.AxisData(ids, label_matrix, Dates.DateTime[])
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

"Add one component axis per label vector of `key` to `reg`. Return the axis keys."
function dataset_axes!(
    reg::AxisRegistry,
    sys,
    key::OptimizationContainerKey{T, U},
    labels::Vector{Vector{String}},
    dataset_name::String,
) where {T, U}
    types = _axis_types(U)
    axis_keys = String[]
    for (i, column) in enumerate(labels)
        type = types[min(i, length(types))]
        push!(axis_keys, register_component_axis!(reg, sys, type, column, dataset_name))
    end
    return axis_keys
end
