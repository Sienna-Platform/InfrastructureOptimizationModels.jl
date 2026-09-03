const MarketModelContainer = Dict{Symbol, DeviceModel}

"""
Abstract supertype of market-model formulations. Concrete formulations are defined by consumer
packages and select how market components clear against settlement balances.
"""
abstract type AbstractMarketModel <: IS.Optimization.AbstractInfrastructureModel end

function _check_market_formulation(::Type{T}) where {T <: AbstractMarketModel}
    if !isconcretetype(T)
        throw(
            ArgumentError(
                "The market model must contain only concrete types, $(T) is an Abstract Type",
            ),
        )
    end
    return nothing
end

"""
Container of the market clearing description: the formulation type, the component models keyed
like device models, and the settlement domain the market's equalities are keyed on.
"""
mutable struct MarketModel{T <: AbstractMarketModel}
    market_component_models::MarketModelContainer
    settlement_domain::Type{<:IS.InfrastructureSystemsType}
    duals::Vector{DataType}
    subsystem::Union{Nothing, String}

    function MarketModel(
        ::Type{T};
        settlement_domain::Type{<:IS.InfrastructureSystemsType},
        duals = Vector{DataType}(),
    ) where {T <: AbstractMarketModel}
        _check_market_formulation(T)
        return new{T}(MarketModelContainer(), settlement_domain, duals, nothing)
    end
end

get_market_component_models(m::MarketModel) = m.market_component_models
get_settlement_domain(m::MarketModel) = m.settlement_domain
get_duals(m::MarketModel) = m.duals
get_subsystem(m::MarketModel) = m.subsystem
