"""
Unit tests for start-up and shut-down cost objective function construction.
Tests the functions in src/objective_function/start_up_shut_down.jl using mock components.
"""

IOM._sos_status(::Type, ::Type{TestDeviceFormulation}) = IOM.SOSStatusVariable.NO_VARIABLE

IOM.objective_function_multiplier(
    ::Type{TestShutDownVariable},
    ::Type{TestDeviceFormulation},
) = 1.0
IOM.objective_function_multiplier(
    ::Type{TestStartVariable},
    ::Type{TestDeviceFormulation},
) = 1.0

# Float64 startup costs just pass through.
IOM.start_up_cost(
    cost::Float64,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Type{TestStartVariable},
    ::Type{TestDeviceFormulation},
) = cost

# AffExpr startup costs pass through (time-variant path produces AffExpr from param * mult).
IOM.start_up_cost(
    cost::JuMP.AffExpr,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::Type{TestStartVariable},
    ::Type{TestDeviceFormulation},
) = cost

# MockTimeSeriesOperationCost is the mock equivalent of MarketBidTimeSeriesCost
IOM._is_time_series_cost(::MockTimeSeriesOperationCost) = true

# Helper to create a MockThermalGen with specified startup/shutdown costs (static)
function make_thermal_with_costs(
    name::String;
    startup_cost::Float64 = 0.0,
    shutdown_cost::Float64 = 0.0,
    must_run::Bool = false,
)
    op_cost = MockOperationCost(0.0, false, 0.0, startup_cost, shutdown_cost)
    return make_mock_thermal(name; operation_cost = op_cost, must_run = must_run)
end

# Helper to create a MockThermalGen with time-series operation cost
function make_thermal_with_ts_costs(name::String; must_run::Bool = false)
    op_cost = MockTimeSeriesOperationCost()
    return make_mock_thermal(name; operation_cost = op_cost, must_run = must_run)
end

# Helper to set up container with variables for mock devices
function setup_startup_shutdown_test_container(
    time_steps::UnitRange{Int},
    devices::Vector{MockThermalGen},
    ::Type{V};
    resolution = Dates.Hour(1),
) where {V <: IOM.VariableType}
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = resolution,
    )
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), MockDeterministic)
    IOM.set_time_steps!(container, time_steps)

    device_names = [get_name(d) for d in devices]
    var_container = IOM.add_variable_container!(
        container,
        V,
        MockThermalGen,
        device_names,
        time_steps,
    )

    jump_model = IOM.get_jump_model(container)
    for name in device_names, t in time_steps
        var_container[name, t] = JuMP.@variable(
            jump_model,
            base_name = "Test_$(name)_$(t)",
        )
    end

    return container
end

# Create a FlattenIteratorWrapper from a vector of mock devices
function make_mock_device_iterator(devices::Vector{MockThermalGen})
    return IS.FlattenIteratorWrapper(MockThermalGen, Vector[devices])
end

@testset "Start-up and Shut-down Cost Objective Functions" begin
    @testset "add_shut_down_cost! adds shutdown cost to objective" begin
        time_steps = 1:3
        shutdown_cost = 50.0
        device = make_thermal_with_costs("gen1"; shutdown_cost = shutdown_cost)
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestShutDownVariable,
        )

        devices_iter = make_mock_device_iterator(devices)

        IOM.add_shut_down_cost!(
            container,
            TestShutDownVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        # Verify shutdown costs are in invariant expression (time-invariant case)
        @test verify_objective_coefficients(
            container,
            TestShutDownVariable,
            MockThermalGen,
            "gen1",
            shutdown_cost;
            variant = false,
        )
    end

    @testset "add_shut_down_cost! skips must_run devices" begin
        time_steps = 1:2
        shutdown_cost = 50.0
        device = make_thermal_with_costs(
            "gen1";
            shutdown_cost = shutdown_cost,
            must_run = true,
        )
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestShutDownVariable,
        )

        devices_iter = make_mock_device_iterator(devices)

        IOM.add_shut_down_cost!(
            container,
            TestShutDownVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        # must_run device should be skipped - no cost terms added
        @test count_objective_terms(container; variant = false) == 0
    end

    @testset "add_shut_down_cost! with zero cost skips device" begin
        time_steps = 1:2
        device = make_thermal_with_costs("gen1"; shutdown_cost = 0.0)
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestShutDownVariable,
        )

        devices_iter = make_mock_device_iterator(devices)

        IOM.add_shut_down_cost!(
            container,
            TestShutDownVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        # Zero cost should be skipped
        @test count_objective_terms(container; variant = false) == 0
    end

    @testset "add_start_up_cost! adds startup cost to objective" begin
        time_steps = 1:3
        startup_cost = 100.0
        device = make_thermal_with_costs("gen1"; startup_cost = startup_cost)
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestStartVariable,
        )

        devices_iter = make_mock_device_iterator(devices)

        IOM.add_start_up_cost!(
            container,
            TestStartVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        # Verify startup costs are in invariant expression (time-invariant case)
        @test verify_objective_coefficients(
            container,
            TestStartVariable,
            MockThermalGen,
            "gen1",
            startup_cost;
            variant = false,
        )
    end

    @testset "add_start_up_cost! skips must_run devices" begin
        time_steps = 1:2
        startup_cost = 100.0
        device = make_thermal_with_costs(
            "gen1";
            startup_cost = startup_cost,
            must_run = true,
        )
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestStartVariable,
        )

        devices_iter = make_mock_device_iterator(devices)

        IOM.add_start_up_cost!(
            container,
            TestStartVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        # must_run device should be skipped - no cost terms added
        @test count_objective_terms(container; variant = false) == 0
    end

    @testset "add_start_up_cost! with zero cost skips device" begin
        time_steps = 1:2
        device = make_thermal_with_costs("gen1"; startup_cost = 0.0)
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestStartVariable,
        )

        devices_iter = make_mock_device_iterator(devices)

        IOM.add_start_up_cost!(
            container,
            TestStartVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        # Zero cost should be skipped
        @test count_objective_terms(container; variant = false) == 0
    end

    @testset "add_start_up_cost! and add_shut_down_cost! with multiple devices" begin
        time_steps = 1:2
        startup1, shutdown1 = 100.0, 50.0
        startup2, shutdown2 = 200.0, 75.0

        device1 = make_thermal_with_costs(
            "gen1";
            startup_cost = startup1,
            shutdown_cost = shutdown1,
        )
        device2 = make_thermal_with_costs(
            "gen2";
            startup_cost = startup2,
            shutdown_cost = shutdown2,
        )
        devices = [device1, device2]

        # Test shutdown costs
        container_sd = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestShutDownVariable,
        )
        devices_iter = make_mock_device_iterator(devices)

        IOM.add_shut_down_cost!(
            container_sd,
            TestShutDownVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        @test verify_objective_coefficients(
            container_sd,
            TestShutDownVariable,
            MockThermalGen,
            "gen1",
            shutdown1;
            variant = false,
        )
        @test verify_objective_coefficients(
            container_sd,
            TestShutDownVariable,
            MockThermalGen,
            "gen2",
            shutdown2;
            variant = false,
        )

        # Test startup costs
        container_su = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestStartVariable,
        )
        devices_iter = make_mock_device_iterator(devices)

        IOM.add_start_up_cost!(
            container_su,
            TestStartVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        @test verify_objective_coefficients(
            container_su,
            TestStartVariable,
            MockThermalGen,
            "gen1",
            startup1;
            variant = false,
        )
        @test verify_objective_coefficients(
            container_su,
            TestStartVariable,
            MockThermalGen,
            "gen2",
            startup2;
            variant = false,
        )
    end

    @testset "add_shut_down_cost! time-variant path" begin
        time_steps = 1:3
        device = make_thermal_with_ts_costs("gen1")
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestShutDownVariable,
        )

        # Set up ShutdownCostParameter with known values
        shutdown_values = [10.0 20.0 30.0]
        add_test_parameter!(
            container,
            IOM.ShutdownCostParameter,
            MockThermalGen,
            ["gen1"],
            time_steps,
            shutdown_values,
        )

        devices_iter = make_mock_device_iterator(devices)

        IOM.add_shut_down_cost!(
            container,
            TestShutDownVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        # Time-variant costs go into variant expression
        @test verify_objective_coefficients(
            container,
            TestShutDownVariable,
            MockThermalGen,
            "gen1",
            shutdown_values[1, :];
            variant = true,
        )
    end

    @testset "add_start_up_cost! time-variant path" begin
        time_steps = 1:3
        device = make_thermal_with_ts_costs("gen1")
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestStartVariable,
        )

        # Set up StartupCostParameter with known values
        startup_values = [50.0 100.0 150.0]
        add_test_parameter!(
            container,
            IOM.StartupCostParameter,
            MockThermalGen,
            ["gen1"],
            time_steps,
            startup_values,
        )

        devices_iter = make_mock_device_iterator(devices)

        IOM.add_start_up_cost!(
            container,
            TestStartVariable,
            devices_iter,
            TestDeviceFormulation,
        )

        # Time-variant costs go into variant expression
        @test verify_objective_coefficients(
            container,
            TestStartVariable,
            MockThermalGen,
            "gen1",
            startup_values[1, :];
            variant = true,
        )
    end
end
