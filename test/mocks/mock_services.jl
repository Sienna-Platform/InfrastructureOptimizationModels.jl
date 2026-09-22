"""
Minimal service mocks.
"""

struct MockUp end
struct MockDown end

struct MockReserve{D} <: IS.InfrastructureSystemsComponent
    name::String
    requirement::Float64
    contributing_devices::Vector{Any}
end

get_name(r::MockReserve) = r.name
get_requirement(r::MockReserve) = r.requirement
