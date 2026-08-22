"""
Minimal time series mocks for testing parameter updates.
"""

using Dates

struct MockDeterministic <: IS.TimeSeriesData{Float64}
    name::String
    data::Vector{Float64}
    resolution::Dates.Period
    initial_timestamp::DateTime
end

struct MockSingleTimeSeries
    name::String
    data::Vector{Float64}
    timestamps::Vector{DateTime}
end

get_name(ts::Union{MockDeterministic, MockSingleTimeSeries}) = ts.name

# Mock components are immutable and hold no time series manager, so `IS.has_time_series`
# can't work off the component itself. Builders that filter devices on time series
# ownership (e.g. the parameterized range constraints) consult this registry instead.
const MOCK_TIME_SERIES_REGISTRY = Dict{String, Set{String}}()

function mock_add_time_series!(component, ts_name::AbstractString)
    push!(get!(MOCK_TIME_SERIES_REGISTRY, get_name(component), Set{String}()), ts_name)
    return
end

mock_clear_time_series!() = empty!(MOCK_TIME_SERIES_REGISTRY)

function IS.has_time_series(
    component::AbstractMockDevice,
    ::Type{MockDeterministic},
    ts_name::AbstractString,
)
    return ts_name in get(MOCK_TIME_SERIES_REGISTRY, get_name(component), Set{String}())
end
