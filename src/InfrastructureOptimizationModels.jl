module InfrastructureOptimizationModels

#################################################################################
# Imports

import DataStructures: OrderedDict, Deque, SortedDict
import Logging
import Serialization
# Modeling Imports
import JuMP
# so that users do not need to import JuMP to use a solver with PowerModels
import JuMP: optimizer_with_attributes
import JuMP.Containers: DenseAxisArray, SparseAxisArray
import MathOptInterface
import LinearAlgebra
import JSON3
import PowerSystems
import InfrastructureSystems
import PowerNetworkMatrices
import PowerNetworkMatrices: PTDF, VirtualPTDF, LODF, VirtualLODF
import InfrastructureSystems: @assert_op, TableFormat, list_recorder_events, get_name
import InfrastructureSystems:
    get_value_curve, get_power_units, get_function_data, get_proportional_term,
    get_quadratic_term, get_fuel_cost

import InfrastructureSystems.Simulation:
    SimulationInfo,
    get_number,
    set_number!,
    get_sequence_uuid,
    set_sequence_uuid!,
    get_run_status,
    set_run_status!

# IS.Optimization imports: base types that remain in InfrastructureSystems
# TODO: some of these are device specific enough to belong in POM.
import InfrastructureSystems.Optimization:
    AbstractOptimizationContainer,
    OptimizationKeyType,
    AbstractModelStoreParams,
    # Key types - imported from IS.Optimization to avoid duplication
    VariableType,
    ConstraintType,
    AuxVariableType,
    ParameterType,
    InitialConditionType,
    ExpressionType,
    RightHandSideParameter,
    ObjectiveFunctionParameter,
    TimeSeriesParameter,
    ConstructStage,
    ArgumentConstructStage,
    ModelConstructStage,
    # Formulation abstract types
    AbstractDeviceFormulation,
    AbstractServiceFormulation,
    AbstractReservesFormulation,
    AbstractThermalFormulation,
    AbstractRenewableFormulation,
    AbstractStorageFormulation,
    AbstractLoadFormulation,
    AbstractHVDCNetworkModel,
    AbstractPowerModel,
    AbstractPowerFlowEvaluationModel,
    AbstractPowerFlowEvaluationData

import InfrastructureSystems:
    @scoped_enum,
    TableFormat,
    get_variables,
    get_parameters,
    get_total_cost,
    get_optimizer_stats,
    get_timestamp,
    write_outputs,
    get_source_data,
    configure_logging,
    strip_module_name,
    to_namedtuple,
    get_uuid,
    compute_file_hash,
    convert_for_path,
    COMPONENT_NAME_DELIMITER,
    # Additional imports needed by core optimization files
    InfrastructureSystemsType,
    InfrastructureSystemsComponent,
    Outputs,
    TimeSeriesCacheKey,
    TimeSeriesCache,
    InvalidValue,
    ConflictingInputsError

# PowerSystems imports
import PowerSystems:
    get_components,
    get_component,
    get_available_components,
    get_available_component,
    get_groups,
    get_available_groups,
    stores_time_series_in_memory,
    get_base_power,
    get_active_power_limits,
    get_start_up,
    get_shut_down,
    get_must_run,
    get_operation_cost
import PowerSystems: StartUpStages

import TimerOutputs

# Base Imports
import Base.getindex
import Base.isempty
import Base.length
import Base.first
import InteractiveUtils: methodswith

# TimeStamp Management Imports
import Dates
import TimeSeries

# I/O Imports
import CSV
import DataFrames
import DataFrames: DataFrame, DataFrameRow, Not, innerjoin, select
import DataFramesMeta: @chain, @orderby, @rename, @select, @subset, @transform
import HDF5
import PrettyTables

################################################################################
# Type Aliases

const PSY = PowerSystems
const POM = InfrastructureOptimizationModels
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const MOPFM = MOI.FileFormats.Model
const PNM = PowerNetworkMatrices
const TS = TimeSeries

################################################################################

using DocStringExtensions

@template DEFAULT = """
                    $(TYPEDSIGNATURES)
                    $(DOCSTRING)
                    """

#################################################################################
# Exports

# Base Models
export DecisionModel
export EmulationModel
export AbstractProblemTemplate
export ServicesModelContainer, DevicesModelContainer, BranchModelContainer
export InitialCondition

# Network Relevant Exports
export NetworkModel
export get_PTDF_matrix, get_LODF_matrix, get_reduce_radial_branches
export get_duals, get_reference_buses, get_subnetworks, get_bus_area_map
export get_power_flow_evaluation, has_subnetworks, get_subsystem
export set_subsystem!, add_dual!
export requires_all_branch_models, supports_branch_filtering, ignores_branch_filtering
export validate_network_model
export validate_available_devices
export BranchReductionOptimizationTracker
export get_variable_dict, get_constraint_dict, get_constraint_map_by_type
export get_number_of_steps, set_number_of_steps!
# Note: Concrete network model types (PTDFPowerModel, CopperPlatePowerModel, etc.)
# are defined in PowerOperationsModels, not IOM.

######## Model Container Types ########
export DeviceModel
export ServiceModel
export FixedOutput
export get_device_cache

# Parameter Container Infrastructure
export ParameterContainer
export ParameterAttributes
export NoAttributes
export TimeSeriesAttributes
export VariableValueAttributes
export CostFunctionAttributes
export EventParametersAttributes
export ValidDataParamEltypes

# Functions
export validate_time_series!
export init_optimization_container!
## Op Model Exports
export get_initial_conditions
export serialize_outputs
export serialize_optimization_model

export get_device_models
export get_branch_models
export get_service_models
export set_device_model!
export set_service_model!
export set_network_model!
export get_network_formulation
export get_hvdc_network_model
export set_hvdc_network_model!

# Extension points for downstream packages (e.g., PowerOperationsModels)
# These functions have fallback implementations in IOM but are meant to be
# extended with device-specific methods in POM
export add_variables!
export add_to_expression!
export add_constant_to_jump_expression!
export add_proportional_to_jump_expression!
export add_linear_to_jump_expression!
# Cost term helpers (generic objective function building blocks)
export add_cost_term_invariant!
export add_cost_term_variant!
export add_pwl_variables_delta!
export add_pwl_linking_constraint!
export add_pwl_normalization_constraint!
export add_pwl_sos2_constraint!
export get_pwl_cost_expression_delta
export process_market_bid_parameters!

## Outputs interfaces
export get_variable_values
export get_dual_values
export get_parameter_values
export get_aux_variable_values
export get_expression_values
export get_timestamps
export get_system
export list_variable_keys
export list_dual_keys
export list_parameter_keys
export list_aux_variable_keys
export list_expression_keys
export list_variable_names
export list_dual_names
export list_parameter_names
export list_aux_variable_names
export list_expression_names
export read_variable
export read_dual
export read_parameter
export read_aux_variable
export read_expression
export read_variables
export read_duals
export read_parameters
export read_aux_variables
export read_expressions
export get_realized_timestamps
export get_problem_base_power
export get_objective_value
export read_optimizer_stats

## Utils Exports
export OptimizationProblemOutputs
export OptimizationProblemOutputsExport
export OptimizerStats
export get_all_constraint_index
export get_all_variable_index
export get_constraint_index
export get_variable_index
export list_recorder_events
export jump_value
export ConstraintBounds
export VariableBounds

# Internal accessors needed by downstream packages
export get_network_model
export get_value
export get_initial_conditions_data
export get_initial_condition_value
export get_objective_expression
export get_formulation
export get_settings
export get_rebuild_model

# Expression infrastructure (needed by add_to_expression.jl implementations)
export get_parameter
export get_parameter_array
export get_network_reduction
export get_multiplier_array
export get_parameter_column_refs
export get_service_name
export get_default_time_series_type
export add_expression_container!

# Initial condition infrastructure (extension points for POM)
export update_initial_conditions!

# Key Types (defined in IOM)
export OptimizationContainerKey
export VariableKey
export ConstraintKey
export ParameterKey
export ExpressionKey
export AuxVarKey

# Abstract Key Types (from InfrastructureSystems.Optimization)
export VariableType
export ConstraintType
export AuxVariableType
export ParameterType
export InitialConditionType
export ExpressionType

# objective_function folder exports
export StartupCostParameter
export ShutdownCostParameter
export OnStatusParameter

# core folder exports
# optimization_container.jl refactor
# add_param_container!
export add_param_container!,
    add_param_container_split_axes!,
    add_param_container_shared_axes!
export remove_undef!
export get_branch_argument_variable_axis

# Bulk-added: symbols used by POM but previously not exported
# Network reduction helpers
export get_branch_argument_constraint_axis, get_reduced_branch_tracker
export search_for_reduced_branch_variable!
export search_for_reduced_branch_parameter!
export search_for_reduced_branch_argument!
export get_branch_argument_parameter_axes
export get_parameter_dict
export get_device_with_time_series
# Container/variable helpers
export add_variable_container!, add_constraint_dual!
export add_to_objective_invariant_expression!, lazy_container_addition!
export get_parameter_multiplier_array, get_aux_variable, get_condition
export supports_milp, get_quadratic_cost_per_system_unit
export check_hvdc_line_limits_unidirectional, check_hvdc_line_limits_consistency
export add_sparse_pwl_interpolation_variables!
export JuMPOrFloat
# Constraint helpers
export add_range_constraints!, add_parameterized_upper_bound_range_constraints
export add_reserve_bound_range_constraints!
export add_semicontinuous_range_constraints!, add_semicontinuous_ramp_constraints!
# Cost helpers
export add_shut_down_cost!, add_start_up_cost!
export add_pwl_term_delta!, add_pwl_constraint_delta!
export add_pwl_term_lambda!, _get_sos_value, _onvar_cost
export uses_commitment_variables
export add_cost_to_expression!
# Duration constraint helpers
export device_duration_compact_retrospective!
export device_duration_parameters!, device_duration_retrospective!
# Ramp helpers
export add_linear_ramp_constraints!
export get_min_max_limits
export AbstractThermalDispatchFormulation, AbstractThermalUnitCommitment
# Service/misc helpers
# NOTE: get_time_series NOT exported — conflicts with IS.get_time_series. Use IOM.get_time_series.
export process_import_export_parameters!, process_market_bid_parameters!
# Extension point functions
export add_service_variables!, requires_initialization
# End bulk-added

# more extension points
export write_outputs!
export built_for_recurrent_solves
export get_incompatible_devices

# Bulk export: symbols POM needs that weren't previously exported
# Core types
export OptimizationContainer, OperationModel, AbstractPowerFlowEvaluationModel
export ArgumentConstructStage, ModelConstructStage
export EmulationModelStore, DeviceModelForBranches
export StartUpStages, SOSStatusVariable
# Parameter types
export FuelCostParameter, VariableValueParameter, FixValueParameter
# Offer curve types (parameter, variable, constraint)
export AbstractCostAtMinParameter,
    IncrementalCostAtMinParameter, DecrementalCostAtMinParameter
export AbstractPiecewiseLinearSlopeParameter,
    IncrementalPiecewiseLinearSlopeParameter, DecrementalPiecewiseLinearSlopeParameter
export AbstractPiecewiseLinearBreakpointParameter,
    IncrementalPiecewiseLinearBreakpointParameter,
    DecrementalPiecewiseLinearBreakpointParameter
export AbstractPiecewiseLinearBlockOffer,
    PiecewiseLinearBlockIncrementalOffer, PiecewiseLinearBlockDecrementalOffer
export AbstractPiecewiseLinearBlockOfferConstraint,
    PiecewiseLinearBlockIncrementalOfferConstraint,
    PiecewiseLinearBlockDecrementalOfferConstraint
# Logging
export LOG_GROUP_BUILD_INITIAL_CONDITIONS,
    LOG_GROUP_COST_FUNCTIONS,
    LOG_GROUP_BRANCH_CONSTRUCTIONS,
    LOG_GROUP_NETWORK_CONSTRUCTION,
    LOG_GROUP_FEEDFORWARDS_CONSTRUCTION,
    LOG_GROUP_SERVICE_CONSTUCTORS,
    LOG_GROUP_OPTIMIZATION_CONTAINER
# Container access
export get_available_components, get_attribute, get_time_steps, get_time_series_names
export get_initial_condition, get_expression, get_variable
export has_container_key, get_constraints, get_constraint, get_aux_variables
export get_optimization_container, get_internal
# Container creation
export add_constraints_container!, add_variable_cost!
# Initial conditions
export add_initial_condition_container!
export has_initial_condition_value, set_ic_quantity!, get_last_recorded_value
export set_initial_conditions_model_container!, get_initial_conditions_model_container
export get_component_type, get_component_name, add_jump_parameter
# Template/model access
export get_use_slacks, get_template, get_model
export get_attributes, get_parameter_column_values
export get_services, get_contributing_devices, get_contributing_devices_map
export set_resolution!, finalize_template!
# JuMP access
export get_jump_model
export _get_breakpoints_for_pwl_function, _add_generic_incremental_interpolation_constraint!
# Cost utilities
export get_proportional_cost_per_system_unit
# Output writing/conversion
export should_write_resulting_value, convert_output_to_natural_units
# End bulk export

export variable_cost

# Standard Expression Types (abstract and concrete)
export SystemBalanceExpressions
export RangeConstraintLBExpressions
export RangeConstraintUBExpressions
export CostExpressions
export PostContingencyExpressions
export ActivePowerBalance
export ReactivePowerBalance
export EmergencyUp
export EmergencyDown
export RawACE
export ProductionCostExpression
export FuelConsumptionExpression
export ActivePowerRangeExpressionLB
export ActivePowerRangeExpressionUB
export PostContingencyBranchFlow
export PostContingencyActivePowerGeneration
export NetActivePower
export DCCurrentBalance
export HVDCPowerBalance

# Standard Variable Types (used in IOM infrastructure code, consumed by POM)
export ActivePowerVariable, ActivePowerInVariable, ActivePowerOutVariable
export PowerAboveMinimumVariable
export OnVariable, StartVariable, StopVariable
export ReservationVariable
export PiecewiseLinearCostVariable
export RateofChangeConstraintSlackUp, RateofChangeConstraintSlackDown
export DCVoltage
# Abstract types needed by POM for type hierarchy
export SparseVariableType, InterpolationVariableType, BinaryInterpolationVariableType

# intended for these to stay internal, but needed for POM due to moving
# add_reserve_range_constraints! into POM (because it needs FooPowerVariableLimitsConstraint)
export UpperBound, LowerBound, BoundDirection, get_bound_direction
export EventParameter

# Abstract types for extensions (from InfrastructureSystems.Optimization)
export AbstractPowerFlowEvaluationData

# Status Enums (from InfrastructureSystems)
export ModelBuildStatus
export RunStatus
export SimulationBuildStatus

# Problem Types
export DecisionProblem
export EmulationProblem
export DefaultDecisionProblem
export DefaultEmulationProblem

# Settings and Data Types
export Settings
export get_warm_start
export get_horizon, get_initial_time, get_optimizer, get_ext, get_interval
export get_check_components, get_initialize_model, get_initialization_file
export get_deserialize_initial_conditions, get_export_pwl_vars
export get_check_numerical_bounds, get_allow_fails
export get_optimizer_solve_log_print, get_calculate_conflict
export get_detailed_optimizer_stats, get_direct_mode_optimizer
export get_store_variable_names, get_export_optimization_model
export use_time_series_cache
export set_horizon!, set_initial_time!, set_warm_start!
export log_values
export InitialConditionsData

# Constants
export COST_EPSILON
export INITIALIZATION_PROBLEM_HORIZON_COUNT

# Re-exports from imports
export optimizer_with_attributes
export PTDF
export VirtualPTDF
export LODF
export VirtualLODF
export get_name
export get_model_base_power
export get_optimizer_stats
export get_timestamps
export get_resolution

export get_contributing_devices
export get_contributing_devices_map
export get_parameter_column_values
export update_container_parameter_values!
export export_outputs
export get_source_data
export set_source_data!

## Note: Concrete PowerModels types (ACPPowerModel, DCPPowerModel, etc.) are now
## defined and exported by PowerOperationsModels, not IOM.

#################################################################################
# Includes
# NOTE: all tracked files are either included here, or have a commented-out include.

# Core optimization types must come first
include("core/optimization_container_types.jl")       # Abstract types (VariableType, etc.)
include("core/definitions.jl")                        # Aliases and enums (needs VariableType)
# SimulationInfo is defined in IS.Simulation
include("core/optimization_container_keys.jl")        # Keys depend on types
include("core/optimization_container_utils.jl")       # key <-> type <-> field correspondences
include("core/parameter_container.jl")                # Parameter container infrastructure
include("core/abstract_model_store.jl")               # Store depends on keys
include("core/optimizer_stats.jl")                    # Stats standalone
include("core/optimization_container_metadata.jl")    # Metadata depends on keys
include("core/optimization_problem_outputs_export.jl") # Export config
include("core/optimization_problem_outputs.jl")       # Outputs depends on all above
include("core/model_internal.jl")                     # Internal state (needs ModelBuildStatus)

include("core/time_series_parameter_types.jl")

# Core components
include("core/operation_model_abstract_types.jl")
include("core/network_reductions.jl")
include("core/service_model.jl")
include("core/device_model.jl")
include("core/network_model.jl")
include("core/initial_conditions.jl")
include("core/settings.jl")
include("core/dataset.jl")
include("core/dataset_container.jl")
include("core/outputs_by_time.jl")

# Order Required
include("operation/problem_template.jl")
include("core/optimization_container.jl")
include("core/dual_processing.jl")
include("core/model_store_params.jl")

# Standard variable and expression types (after OptimizationContainer is defined)
include("core/standard_variables_expressions.jl")

# Common models - extension points for device formulations
include("common_models/interfaces.jl")
include("common_models/add_variable.jl")
include("common_models/add_auxiliary_variable.jl")
include("common_models/add_constraint_dual.jl")
include("common_models/add_jump_expressions.jl")
include("common_models/set_expression.jl")
include("common_models/get_time_series.jl")
# PWL interpolation methods moved to quadratic_approximations/
include("common_models/constraint_helpers.jl")
include("common_models/range_constraint.jl")
include("common_models/duration_constraints.jl")
include("common_models/rateofchange_constraints.jl")

# Objective function implementations
include("objective_function/cost_term_helpers.jl") # generic helpers: add_cost_term_{invariant,variant}!
include("objective_function/common.jl")
include("objective_function/proportional.jl") # add_proportional_cost! and add_proportional_cost_maybe_time_variant!
include("objective_function/start_up_shut_down.jl") # add_{start_up, shut_down}_cost!
# add_variable_cost_to_objective! implementations and that's it (no other exported functions)
# same 5 arguments: container, variable, component, cost_curve, formulation.
include("objective_function/linear_curve.jl")
include("objective_function/quadratic_curve.jl")
include("objective_function/import_export.jl")

# Offer curve types (pure type definitions, no dependencies)
include("objective_function/offer_curve_types.jl")

# Pure PWL formulation math (must come before cost-data-specific files)
include("objective_function/objective_function_pwl_lambda.jl") # lambda/convex combination PWL
include("objective_function/objective_function_pwl_delta.jl")  # delta/incremental block PWL

# Cost-data-specific mapping to PWL formulations
include("objective_function/piecewise_linear.jl")    # CostCurve/FuelCurve → lambda PWL
include("objective_function/value_curve_cost.jl")    # ValueCurve → delta PWL

# Quadratic approximations (PWL via SOS2)
include("quadratic_approximations/common.jl")
include("quadratic_approximations/no_approx.jl")
include("quadratic_approximations/pwl_utils.jl")
include("quadratic_approximations/incremental.jl")
include("quadratic_approximations/solver_sos2.jl")
include("quadratic_approximations/manual_sos2.jl")
include("quadratic_approximations/sawtooth.jl")
include("quadratic_approximations/epigraph.jl")
include("quadratic_approximations/nmdt_common.jl")
include("quadratic_approximations/nmdt.jl")
include("quadratic_approximations/pwmcc_cuts.jl")

# Bilinear approximations (x·y via Bin2/HybS decomposition)
include("bilinear_approximations/mccormick.jl")
include("bilinear_approximations/bin2.jl")
include("bilinear_approximations/no_approx.jl")
include("bilinear_approximations/hybs.jl")
include("bilinear_approximations/nmdt.jl")

# add_param_container! wrappers — must come after piecewise_linear.jl
# (which defines VariableValueParameter and FixValueParameter)
include("common_models/add_param_container.jl")

include("operation/operation_model_interface.jl")
include("operation/decision_model_store.jl")
include("operation/emulation_model_store.jl")
include("operation/store_common.jl")
include("operation/initial_conditions_update_in_memory_store.jl")
include("operation/decision_model.jl")
include("operation/emulation_model.jl")
include("operation/problem_outputs.jl")
include("operation/time_series_interface.jl")
include("operation/optimization_debugging.jl")
include("operation/model_numerical_analysis_utils.jl")

include("initial_conditions/calculate_initial_condition.jl")

# Utils
include("utils/indexing.jl")
include("utils/print_pt_v3.jl")
include("utils/file_utils.jl")
include("utils/logging.jl")
include("utils/dataframes_utils.jl")
include("utils/jump_utils.jl")
include("utils/powersystems_utils.jl")
include("utils/time_series_utils.jl")
include("utils/datetime_utils.jl")
end
