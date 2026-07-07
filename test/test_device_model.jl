"""
Unit tests for DeviceModel struct and related functions.
Uses mock types from mocks/mock_components.jl - no PowerSystems dependency.
"""

using Test
using InfrastructureOptimizationModels

# TestDeviceFormulation, MockThermalGen, MockRenewableGen, MockLoad,
# AbstractMockDevice, AbstractMockGenerator are defined in mocks/mock_components.jl

@testset "DeviceModel" begin
    @testset "Construction with defaults" begin
        model = IOM.DeviceModel(MockThermalGen, TestDeviceFormulation)

        @test IOM.get_component_type(model) == MockThermalGen
        @test IOM.get_formulation(model) == TestDeviceFormulation
        @test isempty(IOM.get_feedforwards(model))
        @test IOM.get_use_slacks(model) == false
        @test isempty(IOM.get_duals(model))
        @test isempty(IOM.get_services(model))
        @test IOM.get_subsystem(model) === nothing
        @test !IOM.has_service_model(model)
    end

    @testset "Construction with custom values" begin
        model = IOM.DeviceModel(
            MockThermalGen,
            TestDeviceFormulation;
            use_slacks = true,
            attributes = Dict{String, Any}("custom_attr" => 42),
        )

        @test IOM.get_use_slacks(model) == true
        @test IOM.get_attribute(model, "custom_attr") == 42
        @test IOM.get_attribute(model, "nonexistent") === nothing
    end

    @testset "Attributes merging" begin
        # Custom attributes should merge with defaults from get_default_attributes
        model = IOM.DeviceModel(
            MockThermalGen,
            TestDeviceFormulation;
            attributes = Dict{String, Any}("my_key" => "my_value"),
        )

        attrs = IOM.get_attributes(model)
        @test attrs isa Dict{String, Any}
        @test attrs["my_key"] == "my_value"
    end

    @testset "Time series names" begin
        model = IOM.DeviceModel(MockThermalGen, TestDeviceFormulation)
        ts_names = IOM.get_time_series_names(model)
        @test ts_names isa Dict
    end

    @testset "Subsystem management" begin
        model = IOM.DeviceModel(MockThermalGen, TestDeviceFormulation)

        @test IOM.get_subsystem(model) === nothing

        IOM.set_subsystem!(model, "subsystem_1")
        @test IOM.get_subsystem(model) == "subsystem_1"
    end

    @testset "get_attribute with nothing model" begin
        @test IOM.get_attribute(nothing, "any_key") === nothing
    end

    @testset "get_services with nothing" begin
        @test IOM.get_services(nothing) === nothing
    end

    @testset "_check_device_formulation rejects abstract types" begin
        # Should reject abstract device type
        @test_throws ArgumentError IOM._check_device_formulation(AbstractMockDevice)
        @test_throws ArgumentError IOM._check_device_formulation(AbstractMockGenerator)

        # Should reject abstract formulation type
        @test_throws ArgumentError IOM._check_device_formulation(
            IOM.AbstractDeviceFormulation,
        )

        # Should accept concrete types
        @test IOM._check_device_formulation(MockThermalGen) === nothing
        @test IOM._check_device_formulation(TestDeviceFormulation) === nothing
    end

    @testset "DeviceModel rejects abstract types in constructor" begin
        # Abstract device type should fail
        @test_throws ArgumentError IOM.DeviceModel(
            AbstractMockDevice,
            TestDeviceFormulation,
        )

        # Abstract formulation type should fail
        @test_throws ArgumentError IOM.DeviceModel(
            MockThermalGen,
            IOM.AbstractDeviceFormulation,
        )
    end

    @testset "set_model!" begin
        dict = Dict{Symbol, Any}()
        model = IOM.DeviceModel(MockThermalGen, TestDeviceFormulation)

        IOM.set_model!(dict, model)

        @test haskey(dict, :MockThermalGen)
        @test dict[:MockThermalGen] === model
    end

    @testset "set_model! warns on overwrite" begin
        dict = Dict{Symbol, Any}()
        model1 = IOM.DeviceModel(MockThermalGen, TestDeviceFormulation)
        model2 = IOM.DeviceModel(MockThermalGen, TestDeviceFormulation)

        IOM.set_model!(dict, model1)

        # Second call should warn about overwriting
        @test_logs (:warn, r"Overwriting.*existing model") IOM.set_model!(dict, model2)
        @test dict[:MockThermalGen] === model2
    end

    @testset "Multiple device types" begin
        # Test with different mock device types
        thermal_model = IOM.DeviceModel(MockThermalGen, TestDeviceFormulation)
        @test IOM.get_component_type(thermal_model) == MockThermalGen

        renewable_model = IOM.DeviceModel(MockRenewableGen, TestDeviceFormulation)
        @test IOM.get_component_type(renewable_model) == MockRenewableGen

        load_model = IOM.DeviceModel(MockLoad, TestDeviceFormulation)
        @test IOM.get_component_type(load_model) == MockLoad
    end

    @testset "FixedOutput formulation" begin
        # FixedOutput is defined in device_model.jl
        model = IOM.DeviceModel(MockThermalGen, IOM.FixedOutput)
        @test IOM.get_formulation(model) == IOM.FixedOutput
    end
end
