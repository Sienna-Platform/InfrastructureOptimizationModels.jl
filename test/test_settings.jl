"""
Unit tests for Settings struct and related functions.
Uses MockSystem from mocks/mock_system.jl and MockMOIOptimizer from mocks/mock_optimizer.jl
"""

using Dates
using Test
using InfrastructureOptimizationModels

# Define PSI alias if not already defined (mock_components.jl defines it)
if !@isdefined(PSI)
    const PSI = InfrastructureOptimizationModels
end

# MockSystem is defined in mocks/mock_system.jl, MockMOIOptimizer in mocks/mock_optimizer.jl
# Both are loaded by InfrastructureOptimizationModelsTests.jl

@testset "Settings" begin
    @testset "Construction with defaults" begin
        sys = MockSystem(100.0, false)
        settings = PSI.Settings(
            sys;
            horizon = Hour(24),
            resolution = Hour(1),
        )

        @test PSI.get_horizon(settings) == Dates.Millisecond(Hour(24))
        @test PSI.get_resolution(settings) == Dates.Millisecond(Hour(1))
        @test PSI.get_warm_start(settings) == true
        @test PSI.get_optimizer(settings) === nothing
        @test PSI.get_direct_mode_optimizer(settings) == false
        @test PSI.get_optimizer_solve_log_print(settings) == false
        @test PSI.get_detailed_optimizer_stats(settings) == false
        @test PSI.get_calculate_conflict(settings) == false
        @test PSI.get_check_components(settings) == true
        @test PSI.get_initialize_model(settings) == true
        @test PSI.get_initialization_file(settings) == ""
        @test PSI.get_deserialize_initial_conditions(settings) == false
        @test PSI.get_export_pwl_vars(settings) == false
        @test PSI.get_allow_fails(settings) == false
        @test PSI.get_rebuild_model(settings) == false
        @test PSI.get_export_optimization_model(settings) ==
              PSI.OptimizationModelExportFormat.NONE
        @test PSI.get_store_variable_names(settings) == false
        @test PSI.get_check_numerical_bounds(settings) == true
        @test PSI.get_ext(settings) isa Dict{String, Any}
    end

    @testset "Construction with custom values" begin
        sys = MockSystem(100.0, false)
        settings = PSI.Settings(
            sys;
            horizon = Hour(48),
            resolution = Minute(30),
            warm_start = false,
            allow_fails = true,
            check_numerical_bounds = false,
            ext = Dict{String, Any}("custom" => 123),
        )

        @test PSI.get_horizon(settings) == Dates.Millisecond(Hour(48))
        @test PSI.get_resolution(settings) == Dates.Millisecond(Minute(30))
        @test PSI.get_warm_start(settings) == false
        @test PSI.get_check_components(settings) == true
        @test PSI.get_allow_fails(settings) == true
        @test PSI.get_check_numerical_bounds(settings) == false
        @test PSI.get_ext(settings)["custom"] == 123
    end

    @testset "Optimizer handling" begin
        sys = MockSystem(100.0, false)

        # Test with nothing (default)
        settings_none = PSI.Settings(sys; horizon = Hour(24), resolution = Hour(1))
        @test PSI.get_optimizer(settings_none) === nothing

        # Test with MOI.OptimizerWithAttributes (passes through directly)
        opt_with_attrs = JuMP.MOI.OptimizerWithAttributes(MockMOIOptimizer)
        settings_owa = PSI.Settings(
            sys;
            horizon = Hour(24),
            resolution = Hour(1),
            optimizer = opt_with_attrs,
        )
        @test PSI.get_optimizer(settings_owa) isa JuMP.MOI.OptimizerWithAttributes

        # Test with DataType: should be coerced to MOI.OptimizerWithAttributes
        settings_dt = PSI.Settings(
            sys;
            horizon = Hour(24),
            resolution = Hour(1),
            optimizer = MockMOIOptimizer,
        )
        @test PSI.get_optimizer(settings_dt) isa JuMP.MOI.OptimizerWithAttributes
    end

    @testset "Time series cache override for in-memory storage" begin
        # When system stores time series in memory, cache size should be overridden to 0
        sys_in_memory = MockSystem(100.0, true)

        settings = PSI.Settings(
            sys_in_memory;
            horizon = Hour(24),
            resolution = Hour(1),
            time_series_cache_size = 1000,
        )

        @test !PSI.use_time_series_cache(settings)
    end

    @testset "Time series cache preserved for non-memory storage" begin
        sys_not_in_memory = MockSystem(100.0, false)

        settings = PSI.Settings(
            sys_not_in_memory;
            horizon = Hour(24),
            resolution = Hour(1),
            time_series_cache_size = 1000,
        )

        @test PSI.use_time_series_cache(settings)
    end

    @testset "Setters" begin
        sys = MockSystem(100.0, false)
        settings = PSI.Settings(sys; horizon = Hour(24), resolution = Hour(1))

        # Test set_horizon!
        PSI.set_horizon!(settings, Hour(48))
        @test PSI.get_horizon(settings) == Dates.Millisecond(Hour(48))

        # Test set_resolution!
        PSI.set_resolution!(settings, Minute(15))
        @test PSI.get_resolution(settings) == Dates.Millisecond(Minute(15))

        # Test set_initial_time!
        new_time = DateTime(2024, 6, 15, 12, 0, 0)
        PSI.set_initial_time!(settings, new_time)
        @test PSI.get_initial_time(settings) == new_time

        # Test set_warm_start!
        PSI.set_warm_start!(settings, false)
        @test PSI.get_warm_start(settings) == false
        PSI.set_warm_start!(settings, true)
        @test PSI.get_warm_start(settings) == true
    end

    @testset "Different time period types" begin
        sys = MockSystem(100.0, false)

        # Test with Hour
        settings_hour = PSI.Settings(sys; horizon = Hour(12), resolution = Hour(1))
        @test PSI.get_horizon(settings_hour) == Dates.Millisecond(Hour(12))

        # Test with Minute
        settings_minute = PSI.Settings(sys; horizon = Minute(360), resolution = Minute(15))
        @test PSI.get_horizon(settings_minute) == Dates.Millisecond(Minute(360))
        @test PSI.get_resolution(settings_minute) == Dates.Millisecond(Minute(15))

        # Test with Second
        settings_second =
            PSI.Settings(sys; horizon = Second(3600), resolution = Second(300))
        @test PSI.get_horizon(settings_second) == Dates.Millisecond(Second(3600))
    end
end
