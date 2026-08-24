"""
Verify mock objects work correctly by instantiating each struct and calling each function.
"""

# MockMOIOptimizer
MockMOIOptimizer()

# MockSystem
sys = MockSystem()
sys = MockSystem(100.0)
sys = MockSystem(100.0, true)
IOM.get_base_power(sys)
IOM.stores_time_series_in_memory(sys)
get_components(MockBus, sys)
add_component!(sys, MockBus("b", 1, :PV))
get_time_series(MockDeterministic, sys, nothing)
add_time_series!(sys, nothing, MockDeterministic("ts", Float64[], Hour(1), DateTime(2024)))

# MockBus
bus = MockBus("bus1", 1, :PV)
get_name(bus)
get_number(bus)
get_bustype(bus)

# MockOperationCost
MockOperationCost(10.0)
MockOperationCost(10.0, true)
MockOperationCost(10.0, false, 5.0)

# MockThermalGen
gen = MockThermalGen("gen1", true, bus, (min = 0.0, max = 100.0))
MockThermalGen("gen2", true, bus, (min = 0.0, max = 100.0), 200.0)
MockThermalGen("gen3", true, bus, (min = 0.0, max = 100.0), 200.0, MockOperationCost(5.0))
get_name(gen)
get_available(gen)
get_bus(gen)
IOM.get_active_power_limits(gen)
IOM.get_base_power(gen)
IOM.get_operation_cost(gen)
IS.get_fuel_cost(gen)

# MockRenewableGen
renewable = MockRenewableGen("wind1", true, bus, 50.0)
get_name(renewable)
get_available(renewable)
get_bus(renewable)
get_rating(renewable)

# MockLoad
load = MockLoad("load1", true, bus, 75.0)
get_name(load)
get_available(load)
get_bus(load)
IOM.get_max_active_power(load)

# MockBranch
bus2 = MockBus("bus2", 2, :PQ)
branch = MockBranch("line1", true, bus, bus2, 100.0)
get_name(branch)
get_available(branch)
get_from_bus(branch)
get_to_bus(branch)
get_rate(branch)

# MockComponentType
MockComponentType()

# TestDeviceFormulation
TestDeviceFormulation

# MockReserve
reserve = MockReserve("reserve", 50.0, [gen])
get_name(reserve)
get_requirement(reserve)

# MockDeterministic
ts_det = MockDeterministic("forecast", rand(24), Hour(1), DateTime(2024, 1, 1))
get_name(ts_det)

# MockSingleTimeSeries
ts_single = MockSingleTimeSeries("actual", rand(24), DateTime[])
get_name(ts_single)

# MockContainer
container = MockContainer()
container.constraints

# Factory functions
make_mock_system(; n_buses = 2, n_gens = 1, n_loads = 1)
make_mock_time_series(; length = 24)
make_mock_thermal("gen")
