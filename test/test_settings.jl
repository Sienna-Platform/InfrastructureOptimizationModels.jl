"""
Unit tests for Settings struct and related functions.
Uses MockSystem from mocks/mock_system.jl and MockMOIOptimizer from mocks/mock_optimizer.jl
"""

using Dates
using Test
using InfrastructureOptimizationModels

# MockSystem is defined in mocks/mock_system.jl, MockMOIOptimizer in mocks/mock_optimizer.jl
# Both are loaded by InfrastructureOptimizationModelsTests.jl

@testset "Settings" begin
    @testset "Construction with defaults" begin
        sys = MockSystem(100.0, false)
        settings = IOM.Settings(
            sys;
            horizon = Hour(24),
            resolution = Hour(1),
        )

        @test IOM.get_horizon(settings) == Dates.Millisecond(Hour(24))
        @test IOM.get_resolution(settings) == Dates.Millisecond(Hour(1))
        @test IOM.get_warm_start(settings) == true
        @test IOM.get_optimizer(settings) === nothing
        @test IOM.get_direct_mode_optimizer(settings) == false
        @test IOM.get_optimizer_solve_log_print(settings) == false
        @test IOM.get_detailed_optimizer_stats(settings) == false
        @test IOM.get_calculate_conflict(settings) == false
        @test IOM.get_check_components(settings) == true
        @test IOM.get_initialize_model(settings) == true
        @test IOM.get_initialization_file(settings) == ""
        @test IOM.get_deserialize_initial_conditions(settings) == false
        @test IOM.get_export_pwl_vars(settings) == false
        @test IOM.get_allow_fails(settings) == false
        @test IOM.get_rebuild_model(settings) == false
        @test IOM.get_export_optimization_model(settings) ==
              IOM.OptimizationModelExportFormat.NONE
        @test IOM.get_store_variable_names(settings) == false
        @test IOM.get_check_numerical_bounds(settings) == true
        @test IOM.get_ext(settings) isa Dict{String, Any}
    end

    @testset "Construction with custom values" begin
        sys = MockSystem(100.0, false)
        settings = IOM.Settings(
            sys;
            horizon = Hour(48),
            resolution = Minute(30),
            warm_start = false,
            allow_fails = true,
            check_numerical_bounds = false,
            ext = Dict{String, Any}("custom" => 123),
        )

        @test IOM.get_horizon(settings) == Dates.Millisecond(Hour(48))
        @test IOM.get_resolution(settings) == Dates.Millisecond(Minute(30))
        @test IOM.get_warm_start(settings) == false
        @test IOM.get_check_components(settings) == true
        @test IOM.get_allow_fails(settings) == true
        @test IOM.get_check_numerical_bounds(settings) == false
        @test IOM.get_ext(settings)["custom"] == 123
    end

    @testset "Optimizer handling" begin
        sys = MockSystem(100.0, false)

        # Test with nothing (default)
        settings_none = IOM.Settings(sys; horizon = Hour(24), resolution = Hour(1))
        @test IOM.get_optimizer(settings_none) === nothing

        # Test with MOI.OptimizerWithAttributes (passes through directly)
        opt_with_attrs = JuMP.MOI.OptimizerWithAttributes(MockMOIOptimizer)
        settings_owa = IOM.Settings(
            sys;
            horizon = Hour(24),
            resolution = Hour(1),
            optimizer = opt_with_attrs,
        )
        @test IOM.get_optimizer(settings_owa) isa JuMP.MOI.OptimizerWithAttributes

        # Test with DataType: should be coerced to MOI.OptimizerWithAttributes
        settings_dt = IOM.Settings(
            sys;
            horizon = Hour(24),
            resolution = Hour(1),
            optimizer = MockMOIOptimizer,
        )
        @test IOM.get_optimizer(settings_dt) isa JuMP.MOI.OptimizerWithAttributes
    end

    @testset "Time series cache override for in-memory storage" begin
        # When system stores time series in memory, cache size should be overridden to 0
        sys_in_memory = MockSystem(100.0, true)

        settings = IOM.Settings(
            sys_in_memory;
            horizon = Hour(24),
            resolution = Hour(1),
            time_series_cache_size = 1000,
        )

        @test !IOM.use_time_series_cache(settings)
    end

    @testset "Time series cache preserved for non-memory storage" begin
        sys_not_in_memory = MockSystem(100.0, false)

        settings = IOM.Settings(
            sys_not_in_memory;
            horizon = Hour(24),
            resolution = Hour(1),
            time_series_cache_size = 1000,
        )

        @test IOM.use_time_series_cache(settings)
    end

    @testset "Setters" begin
        sys = MockSystem(100.0, false)
        settings = IOM.Settings(sys; horizon = Hour(24), resolution = Hour(1))

        # Test set_horizon!
        IOM.set_horizon!(settings, Hour(48))
        @test IOM.get_horizon(settings) == Dates.Millisecond(Hour(48))

        # Test set_resolution!
        IOM.set_resolution!(settings, Minute(15))
        @test IOM.get_resolution(settings) == Dates.Millisecond(Minute(15))

        # Test set_initial_time!
        new_time = DateTime(2024, 6, 15, 12, 0, 0)
        IOM.set_initial_time!(settings, new_time)
        @test IOM.get_initial_time(settings) == new_time

        # Test set_warm_start!
        IOM.set_warm_start!(settings, false)
        @test IOM.get_warm_start(settings) == false
        IOM.set_warm_start!(settings, true)
        @test IOM.get_warm_start(settings) == true
    end

    @testset "Different time period types" begin
        sys = MockSystem(100.0, false)

        # Test with Hour
        settings_hour = IOM.Settings(sys; horizon = Hour(12), resolution = Hour(1))
        @test IOM.get_horizon(settings_hour) == Dates.Millisecond(Hour(12))

        # Test with Minute
        settings_minute = IOM.Settings(sys; horizon = Minute(360), resolution = Minute(15))
        @test IOM.get_horizon(settings_minute) == Dates.Millisecond(Minute(360))
        @test IOM.get_resolution(settings_minute) == Dates.Millisecond(Minute(15))

        # Test with Second
        settings_second =
            IOM.Settings(sys; horizon = Second(3600), resolution = Second(300))
        @test IOM.get_horizon(settings_second) == Dates.Millisecond(Second(3600))
    end
end
