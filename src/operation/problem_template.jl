
const DevicesModelContainer = Dict{Symbol, DeviceModel}
const ServicesModelContainer = Dict{Tuple{String, Symbol}, ServiceModel}

abstract type AbstractProblemTemplate end

# Interface defaults — concrete implementations provided by downstream packages
function get_device_models(template::AbstractProblemTemplate)
    throw(
        ArgumentError(
            "get_device_models is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `get_device_models(::YourTemplateType)` " *
            "returning a `DevicesModelContainer`.",
        ),
    )
end

function get_branch_models(template::AbstractProblemTemplate)
    throw(
        ArgumentError(
            "get_branch_models is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `get_branch_models(::YourTemplateType)` " *
            "returning the branch models for the template.",
        ),
    )
end

function get_service_models(template::AbstractProblemTemplate)
    throw(
        ArgumentError(
            "get_service_models is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `get_service_models(::YourTemplateType)` " *
            "returning a `ServicesModelContainer`.",
        ),
    )
end

function get_network_model(template::AbstractProblemTemplate)
    throw(
        ArgumentError(
            "get_network_model is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `get_network_model(::YourTemplateType)` " *
            "returning the network model for the template.",
        ),
    )
end

function get_network_formulation(template::AbstractProblemTemplate)
    throw(
        ArgumentError(
            "get_network_formulation is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `get_network_formulation(::YourTemplateType)` " *
            "returning the network formulation.",
        ),
    )
end

function get_hvdc_network_model(template::AbstractProblemTemplate)
    throw(
        ArgumentError(
            "get_hvdc_network_model is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `get_hvdc_network_model(::YourTemplateType)` " *
            "returning the HVDC network model for the template, or return `nothing` if not applicable.",
        ),
    )
end

function get_component_types(template::AbstractProblemTemplate)
    throw(
        ArgumentError(
            "get_component_types is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `get_component_types(::YourTemplateType)` " *
            "returning the collection of component types used by the template.",
        ),
    )
end

function get_model(template::AbstractProblemTemplate)
    throw(
        ArgumentError(
            "get_model is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `get_model(::YourTemplateType)` " *
            "returning the underlying optimization or simulation model.",
        ),
    )
end

function set_network_model!(template::AbstractProblemTemplate, args...)
    throw(
        ArgumentError(
            "set_network_model! is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `set_network_model!(::YourTemplateType, network_model)` " *
            "to attach the network model to the template.",
        ),
    )
end

function set_hvdc_network_model!(template::AbstractProblemTemplate, args...)
    throw(
        ArgumentError(
            "set_hvdc_network_model! is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `set_hvdc_network_model!(::YourTemplateType, hvdc_network_model)` " *
            "to attach the HVDC network model to the template.",
        ),
    )
end

function set_device_model!(template::AbstractProblemTemplate, args...)
    throw(
        ArgumentError(
            "set_device_model! is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `set_device_model!(::YourTemplateType, device_model)` " *
            "to attach or update a device model for the template.",
        ),
    )
end

function set_service_model!(template::AbstractProblemTemplate, args...)
    throw(
        ArgumentError(
            "set_service_model! is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `set_service_model!(::YourTemplateType, service_model)` " *
            "to attach or update a service model for the template.",
        ),
    )
end

function finalize_template!(template::AbstractProblemTemplate, args...)
    throw(
        ArgumentError(
            "finalize_template! is not implemented for $(typeof(template)). " *
            "Downstream packages must implement `finalize_template!(::YourTemplateType, ...)` " *
            "to finalize and validate the problem template before use.",
        ),
    )
end

# Deep-copy a template while sharing the network model's PNM matrices by reference:
# their solver caches hold raw factorization handles and deliberately error on deepcopy
# (PNM #312). The matrices are read-only inputs, so sharing is safe.
function _deepcopy_template(template::AbstractProblemTemplate)
    network_model = get_network_model(template)
    network_model === nothing && return deepcopy(template)
    ptdf = network_model.PTDF_matrix
    modf = network_model.MODF_matrix
    network_model.PTDF_matrix = nothing
    network_model.MODF_matrix = nothing
    template_ = deepcopy(template)
    network_model.PTDF_matrix = ptdf
    network_model.MODF_matrix = modf
    copied_network_model = get_network_model(template_)
    copied_network_model.PTDF_matrix = ptdf
    copied_network_model.MODF_matrix = modf
    return template_
end

"""
Return the set of device types whose formulation is FixedOutput (incompatible with
service provision).
"""
function get_incompatible_devices(devices_template::Dict)
    incompatible_device_types = Set{DataType}()
    for model in values(devices_template)
        formulation = get_formulation(model)
        if formulation == FixedOutput
            if !isempty(get_services(model))
                @info "$(formulation) for $(get_component_type(model)) is not compatible with the provision of reserve services"
            end
            push!(incompatible_device_types, get_component_type(model))
        end
    end
    return incompatible_device_types
end
