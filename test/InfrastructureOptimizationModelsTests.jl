module InfrastructureOptimizationModelsTests

#=
Testing Strategy:
- Lightweight tests run first using only mock objects (no PowerSystems types used)
- Then tests that require PowerSystems types run
- All heavy dependencies are loaded at module level (Julia requirement)
  but tests are organized to run lightweight ones first
=#

using Test
using InfrastructureOptimizationModels
using Logging
using DataFrames
import DataFramesMeta: @rsubset
using Dates
using JuMP
import JuMP.Containers: DenseAxisArray, SparseAxisArray
using HiGHS

# Import InfrastructureSystems for logging utilities
using InfrastructureSystems
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization
const IOM = InfrastructureOptimizationModels

# Test directory path for includes
const TEST_DIR = @__DIR__

# Load mock infrastructure (lightweight, no PowerSystems dependency)
include(joinpath(TEST_DIR, "mocks/mock_optimizer.jl"))
include(joinpath(TEST_DIR, "mocks/mock_system.jl"))
include(joinpath(TEST_DIR, "mocks/mock_components.jl"))
include(joinpath(TEST_DIR, "mocks/mock_time_series.jl"))
include(joinpath(TEST_DIR, "mocks/mock_services.jl"))
include(joinpath(TEST_DIR, "mocks/mock_container.jl"))
include(joinpath(TEST_DIR, "mocks/constructors.jl"))
include(joinpath(TEST_DIR, "test_utils/test_types.jl"))
include(joinpath(TEST_DIR, "test_utils/objective_function_helpers.jl"))
include(joinpath(TEST_DIR, "test_utils/qa_bilinear_helpers.jl"))

# Environment flags for test selection
const RUN_UNIT_TESTS = get(ENV, "IOM_RUN_UNIT_TESTS", "true") == "true"

include(joinpath(TEST_DIR, "test_utils/solver_definitions.jl"))

const LOG_FILE = "power-optimization-models-test.log"

# Logging configuration
function setup_logging()
    logging_config_filename = get(ENV, "SIIP_LOGGING_CONFIG", nothing)
    if logging_config_filename !== nothing
        config = IS.LoggingConfiguration(logging_config_filename)
    else
        config = IS.LoggingConfiguration(;
            filename = LOG_FILE,
            file_level = Logging.Info,
            console_level = Logging.Error,
        )
    end
    return config
end

# Main test execution
function run_tests()
    config = setup_logging()
    console_logger = ConsoleLogger(config.console_stream, config.console_level)

    IS.open_file_logger(LOG_FILE, config.file_level) do file_logger
        levels = (Logging.Info, Logging.Warn, Logging.Error)
        multi_logger =
            IS.MultiLogger([console_logger, file_logger], IS.LogEventTracker(levels))
        global_logger(multi_logger)

        if !isempty(config.group_levels)
            IS.set_group_levels!(multi_logger, config.group_levels)
        end

        @info "Running InfrastructureOptimizationModels.jl tests"
        @info "Unit tests: $RUN_UNIT_TESTS"

        if RUN_UNIT_TESTS
            @info "Starting unit tests..."
            @time @testset "InfrastructureOptimizationModels Unit Tests" begin
                #=
                ============================================================================
                LIGHTWEIGHT TESTS (mocks only, no PSY types)
                ============================================================================
                =#
                @testset "Lightweight Tests (mocks only)" begin
                    @info "Running lightweight tests..."

                    # --- core/ subfolder ---
                    # abstract_model_store.jl: not worth testing - abstract type only
                    include(joinpath(TEST_DIR, "test_dataset_container.jl"))
                    # TODO dataset.jl
                    # definitions.jl: no need for tests
                    include(joinpath(TEST_DIR, "test_device_model.jl"))
                    # TODO: initial_conditions.jl
                    include(joinpath(TEST_DIR, "test_model_internal.jl"))
                    # model_store_params.jl: low-complexity
                    # TODO: network_model.jl
                    # TODO: network_reductions.jl
                    # operation_model_abstract_types.jl: low complexity
                    include(joinpath(TEST_DIR, "test_optimization_container_keys.jl"))
                    include(joinpath(TEST_DIR, "test_optimization_container_metadata.jl"))
                    # optimization_container_types.jl: no need for tests
                    include(joinpath(TEST_DIR, "test_optimization_container.jl"))
                    include(joinpath(TEST_DIR, "test_network_model.jl"))
                    include(joinpath(TEST_DIR, "test_external_evaluation.jl"))
                    # optimization_problem_outputs_export.jl: low-complexity
                    include(joinpath(TEST_DIR, "test_optimization_outputs.jl"))
                    include(joinpath(TEST_DIR, "test_optimizer_stats.jl"))
                    # parameter_container.jl: low-complexity
                    # TODO outputs_by_time.jl
                    # TODO service_model.jl
                    include(joinpath(TEST_DIR, "test_settings.jl"))
                    # standard_variables_expressions.jl: low complexity
                    # time_series_parameter_types.jl: low complexity

                    # --- objective_function/ subfolder ---
                    # import_export.jl: commented out
                    include(joinpath(TEST_DIR, "test_cost_term_helpers.jl"))
                    include(joinpath(TEST_DIR, "test_linear_curve.jl"))
                    # value_curve_cost.jl: needs more work
                    include(joinpath(TEST_DIR, "test_piecewise_linear.jl"))
                    include(joinpath(TEST_DIR, "test_proportional.jl"))
                    include(joinpath(TEST_DIR, "test_quadratic_curve.jl"))
                    include(joinpath(TEST_DIR, "test_start_up_shut_down.jl"))
                    include(joinpath(TEST_DIR, "test_ts_value_curve_objective.jl"))
                    include(joinpath(TEST_DIR, "test_tranche_axis_helpers.jl"))

                    # --- common_models/, utils/, initial_conditions/ ---
                    # TODO tests?
                    include(joinpath(TEST_DIR, "test_jump_utils.jl"))
                    include(joinpath(TEST_DIR, "test_pwl_methods.jl"))
                    include(joinpath(TEST_DIR, "test_ramp_constraints.jl"))
                    include(joinpath(TEST_DIR, "test_duration_constraints.jl"))
                    include(joinpath(TEST_DIR, "test_emulation_model_store.jl"))

                    # --- quadratic_approximations/ subfolder ---
                    include(joinpath(TEST_DIR, "test_quadratic_approximations.jl"))
                    include(joinpath(TEST_DIR, "test_bilinear_approximations.jl"))
                    include(joinpath(TEST_DIR, "test_hybs_approximations.jl"))
                    include(joinpath(TEST_DIR, "test_nmdt_approximations.jl"))
                    include(joinpath(TEST_DIR, "test_tolerance_dispatch.jl"))
                end

                #=
                ============================================================================
                BROKEN/NEEDS-WORK TEST FILES (not included — rewrite with mocks)
                ============================================================================
                - test_basic_model_structs.jl, test_model_decision.jl, test_model_emulation.jl:
                  previously used PSY/PowerModels types; rewrite with mocks before re-enabling.
                - test_model_store.jl, test_offer_curve_cost.jl: PSY/PSB-backed integration tests
                  removed when IOM dropped PSY/PSB dependencies; rewrite with mocks.
                ============================================================================
                =#
            end
        end

        @test length(IS.get_log_events(multi_logger.tracker, Logging.Error)) == 0
        @info IS.report_log_summary(multi_logger)
    end
end

end # module InfrastructureOptimizationModelsTests
