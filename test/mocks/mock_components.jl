"""
Minimal mock components that satisfy PowerSystems device interfaces.
Each mock is ~20 lines and implements only get_name, get_available, etc.

These types can be used:
1. As instance types (creating MockThermalGen instances)
2. As type parameters for DeviceModel{D, B} (replacing PSY.ThermalStandard etc.)
3. As type parameters for container keys (VariableKey, ConstraintKey, etc.)
"""

using InfrastructureOptimizationModels
using InfrastructureSystems
const PSI = InfrastructureOptimizationModels
const IS = InfrastructureSystems

# Mock formulation type for testing DeviceModel
struct TestDeviceFormulation <: PSI.AbstractDeviceFormulation end
struct TestPowerModel <: IS.Optimization.AbstractPowerModel end

# Mock operation costs for testing objective function construction.
# Mirrors the PSY pattern: separate static and time-series types.

"Static mock cost — all fields are scalars."
struct MockOperationCost <: IS.DeviceParameter
    proportional_term::Float64
    is_time_variant::Bool
    fuel_cost::Float64
    start_up::Float64
    shut_down::Float64
end

MockOperationCost(proportional_term::Float64) =
    MockOperationCost(proportional_term, false, 0.0, 0.0, 0.0)
MockOperationCost(proportional_term::Float64, is_time_variant::Bool) =
    MockOperationCost(proportional_term, is_time_variant, 0.0, 0.0, 0.0)
MockOperationCost(proportional_term::Float64, is_time_variant::Bool, fuel_cost::Float64) =
    MockOperationCost(proportional_term, is_time_variant, fuel_cost, 0.0, 0.0)

IOM.get_start_up(c::MockOperationCost) = c.start_up
IOM.get_shut_down(c::MockOperationCost) = c.shut_down

"""
Time-series mock cost, paralleling PSY.MarketBidTimeSeriesCost. Has no fields because all
cost data (startup, shutdown, offer curves) lives in parameter containers populated
externally — by POM in real use, or by `add_test_parameter!` in the tests.
"""
struct MockTimeSeriesOperationCost <: IS.DeviceParameter end

# Startup/shutdown values aren't stored on the cost object; they live in parameter containers.
# Return sentinel values that would error if accidentally used as costs.
IOM.get_start_up(::MockTimeSeriesOperationCost) =
    error(
        "MockTimeSeriesOperationCost: start_up should be read from parameters, not the cost object",
    )
IOM.get_shut_down(::MockTimeSeriesOperationCost) =
    error(
        "MockTimeSeriesOperationCost: shut_down should be read from parameters, not the cost object",
    )

# Abstract mock device type for testing rejection of abstract types in DeviceModel
# Subtypes IS.InfrastructureSystemsComponent so they work with DeviceModel and container keys
abstract type AbstractMockDevice <: IS.InfrastructureSystemsComponent end
abstract type AbstractMockGenerator <: AbstractMockDevice end

# Mock Bus
struct MockBus
    name::String
    number::Int
    bustype::Symbol
end

get_name(b::MockBus) = b.name
get_number(b::MockBus) = b.number
get_bustype(b::MockBus) = b.bustype

const MockOperationCostTypes = Union{MockOperationCost, MockTimeSeriesOperationCost}

# Mock Thermal Generator
struct MockThermalGen <: AbstractMockGenerator
    name::String
    available::Bool
    bus::MockBus
    active_power_limits::NamedTuple{(:min, :max), Tuple{Float64, Float64}}
    base_power::Float64
    operation_cost::MockOperationCostTypes
    must_run::Bool
end

# Constructor with default base_power and no operation cost for backward compatibility
MockThermalGen(name, available, bus, limits) =
    MockThermalGen(name, available, bus, limits, 100.0, MockOperationCost(0.0), false)
MockThermalGen(name, available, bus, limits, base_power) =
    MockThermalGen(name, available, bus, limits, base_power, MockOperationCost(0.0), false)
MockThermalGen(name, available, bus, limits, base_power, operation_cost) =
    MockThermalGen(name, available, bus, limits, base_power, operation_cost, false)

get_name(g::MockThermalGen) = g.name
get_available(g::MockThermalGen) = g.available
get_bus(g::MockThermalGen) = g.bus
IOM.get_active_power_limits(g::MockThermalGen) = g.active_power_limits
IOM.get_base_power(g::MockThermalGen) = g.base_power
IOM.get_operation_cost(g::MockThermalGen) = g.operation_cost
IOM.get_must_run(g::MockThermalGen) = g.must_run
IS.get_fuel_cost(g::MockThermalGen) = _mock_fuel_cost(g.operation_cost)
_mock_fuel_cost(c::MockOperationCost) = c.fuel_cost
_mock_fuel_cost(::MockTimeSeriesOperationCost) = 0.0

# Mock Renewable Generator
struct MockRenewableGen <: AbstractMockGenerator
    name::String
    available::Bool
    bus::MockBus
    rating::Float64
end

get_name(r::MockRenewableGen) = r.name
get_available(r::MockRenewableGen) = r.available
get_bus(r::MockRenewableGen) = r.bus
get_rating(r::MockRenewableGen) = r.rating

# Mock Load
struct MockLoad <: AbstractMockDevice
    name::String
    available::Bool
    bus::MockBus
    max_active_power::Float64
end

get_name(l::MockLoad) = l.name
get_available(l::MockLoad) = l.available
get_bus(l::MockLoad) = l.bus
get_max_active_power(l::MockLoad) = l.max_active_power

# Mock Branch
struct MockBranch <: AbstractMockDevice
    name::String
    available::Bool
    from_bus::MockBus
    to_bus::MockBus
    rating::Float64
end

get_name(b::MockBranch) = b.name
get_available(b::MockBranch) = b.available
get_from_bus(b::MockBranch) = b.from_bus
get_to_bus(b::MockBranch) = b.to_bus
get_rate(b::MockBranch) = b.rating

# Mock component type for use as type parameter in container keys
# This replaces PSY.ThermalStandard etc. in tests that don't need real PSY types
# Subtypes IS.InfrastructureSystemsComponent so it works with VariableKey, ConstraintKey, etc.
struct MockComponentType <: IS.InfrastructureSystemsComponent end

# Structures for the network problem
struct MockNetworkNode <: IS.InfrastructureSystemsComponent
    name::String
    loss::Vector{Float64}
    i_min::Float64
    i_max::Float64
    v_min::Float64
    v_max::Float64
end
get_name(n::MockNetworkNode) = n.name

struct MockVoltageVariable <: IOM.VariableType end
struct MockCurrentVariable <: IOM.VariableType end

struct MockPowerRangeConstraint <: IOM.ConstraintType end

IOM.get_variable_binary(
    ::Type{ActivePowerVariable},
    ::Type{MockNetworkNode},
    ::Type{TestDeviceFormulation},
) = false
IOM.get_variable_binary(
    ::Type{MockVoltageVariable},
    ::Type{MockNetworkNode},
    ::Type{TestDeviceFormulation},
) = false
IOM.get_variable_binary(
    ::Type{MockCurrentVariable},
    ::Type{MockNetworkNode},
    ::Type{TestDeviceFormulation},
) = false

IOM.get_variable_lower_bound(
    ::Type{MockVoltageVariable},
    n::MockNetworkNode,
    ::Type{TestDeviceFormulation},
) = n.v_min
IOM.get_variable_upper_bound(
    ::Type{MockVoltageVariable},
    n::MockNetworkNode,
    ::Type{TestDeviceFormulation},
) = n.v_max

IOM.get_variable_lower_bound(
    ::Type{MockCurrentVariable},
    n::MockNetworkNode,
    ::Type{TestDeviceFormulation},
) = n.i_min
IOM.get_variable_upper_bound(
    ::Type{MockCurrentVariable},
    n::MockNetworkNode,
    ::Type{TestDeviceFormulation},
) = n.i_max

IOM.get_min_max_limits(
    ::MockNetworkNode,
    ::Type{MockPowerRangeConstraint},
    ::Type{TestDeviceFormulation},
) = (min = 0.0, max = 1.5)
