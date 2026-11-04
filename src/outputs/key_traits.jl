# Output unit traits of the key types that IOM defines.
# Unit rules: design section 5.6 and 8.2; allowed natural units: SiennaSchemas/Core/units.json.

# Variables

IS.Optimization.output_unit_system(::Type{ActivePowerInVariable}) = IS.SU
IS.Optimization.quantity_kind(::Type{ActivePowerInVariable}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{ActivePowerOutVariable}) = IS.SU
IS.Optimization.quantity_kind(::Type{ActivePowerOutVariable}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{ActivePowerVariable}) = IS.SU
IS.Optimization.quantity_kind(::Type{ActivePowerVariable}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{PowerAboveMinimumVariable}) = IS.SU
IS.Optimization.quantity_kind(::Type{PowerAboveMinimumVariable}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{RateofChangeConstraintSlackDown}) = IS.SU
IS.Optimization.quantity_kind(::Type{RateofChangeConstraintSlackDown}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{RateofChangeConstraintSlackUp}) = IS.SU
IS.Optimization.quantity_kind(::Type{RateofChangeConstraintSlackUp}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{DCVoltage}) = IS.CU
IS.Optimization.quantity_kind(::Type{DCVoltage}) = "Voltage"
IS.Optimization.output_unit_system(::Type{BilinearProductVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{BilinearProductVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{BilinearProductVariable}) = "1"
IS.Optimization.output_unit_system(::Type{EpigraphVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{EpigraphVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{EpigraphVariable}) = "1"
IS.Optimization.output_unit_system(::Type{ManualSOS2BinaryVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{ManualSOS2BinaryVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{ManualSOS2BinaryVariable}) = "1"
IS.Optimization.output_unit_system(::Type{NMDTBinaryContinuousProductVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{NMDTBinaryContinuousProductVariable}) =
    "Dimensionless"
IS.Optimization.natural_units(::Type{NMDTBinaryContinuousProductVariable}) = "1"
IS.Optimization.output_unit_system(::Type{NMDTBinaryVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{NMDTBinaryVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{NMDTBinaryVariable}) = "1"
IS.Optimization.output_unit_system(::Type{NMDTResidualProductVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{NMDTResidualProductVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{NMDTResidualProductVariable}) = "1"
IS.Optimization.output_unit_system(::Type{NMDTResidualVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{NMDTResidualVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{NMDTResidualVariable}) = "1"
IS.Optimization.output_unit_system(::Type{OnVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{OnVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{OnVariable}) = "1"
IS.Optimization.output_unit_system(::Type{PiecewiseMcCormickBinary}) = IS.NU
IS.Optimization.quantity_kind(::Type{PiecewiseMcCormickBinary}) = "Dimensionless"
IS.Optimization.natural_units(::Type{PiecewiseMcCormickBinary}) = "1"
IS.Optimization.output_unit_system(::Type{PiecewiseMcCormickDisaggregated}) = IS.NU
IS.Optimization.quantity_kind(::Type{PiecewiseMcCormickDisaggregated}) = "Dimensionless"
IS.Optimization.natural_units(::Type{PiecewiseMcCormickDisaggregated}) = "1"
IS.Optimization.output_unit_system(::Type{QuadraticVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{QuadraticVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{QuadraticVariable}) = "1"
IS.Optimization.output_unit_system(::Type{ReservationVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{ReservationVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{ReservationVariable}) = "1"
IS.Optimization.output_unit_system(::Type{SawtoothAuxVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{SawtoothAuxVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{SawtoothAuxVariable}) = "1"
IS.Optimization.output_unit_system(::Type{SawtoothBinaryVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{SawtoothBinaryVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{SawtoothBinaryVariable}) = "1"
IS.Optimization.output_unit_system(::Type{SawtoothTightenedVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{SawtoothTightenedVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{SawtoothTightenedVariable}) = "1"
IS.Optimization.output_unit_system(::Type{StartVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{StartVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{StartVariable}) = "1"
IS.Optimization.output_unit_system(::Type{StopVariable}) = IS.NU
IS.Optimization.quantity_kind(::Type{StopVariable}) = "Dimensionless"
IS.Optimization.natural_units(::Type{StopVariable}) = "1"

# Parameters

IS.Optimization.output_unit_system(::Type{ActivePowerInTimeSeriesParameter}) = IS.SU
IS.Optimization.quantity_kind(::Type{ActivePowerInTimeSeriesParameter}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{ActivePowerOutTimeSeriesParameter}) = IS.SU
IS.Optimization.quantity_kind(::Type{ActivePowerOutTimeSeriesParameter}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{ActivePowerTimeSeriesParameter}) = IS.SU
IS.Optimization.quantity_kind(::Type{ActivePowerTimeSeriesParameter}) = "ActivePower"
IS.Optimization.output_unit_system(
    ::Type{DecrementalPiecewiseLinearBreakpointParameter},
) = IS.SU
IS.Optimization.quantity_kind(::Type{DecrementalPiecewiseLinearBreakpointParameter}) =
    "ActivePower"
IS.Optimization.output_unit_system(
    ::Type{IncrementalPiecewiseLinearBreakpointParameter},
) = IS.SU
IS.Optimization.quantity_kind(::Type{IncrementalPiecewiseLinearBreakpointParameter}) =
    "ActivePower"
IS.Optimization.output_unit_system(::Type{RequirementTimeSeriesParameter}) = IS.SU
IS.Optimization.quantity_kind(::Type{RequirementTimeSeriesParameter}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{ReactivePowerTimeSeriesParameter}) = IS.SU
IS.Optimization.quantity_kind(::Type{ReactivePowerTimeSeriesParameter}) =
    "ReactivePower"
IS.Optimization.output_unit_system(::Type{FixValueParameter}) = IS.NU
IS.Optimization.quantity_kind(::Type{FixValueParameter}) = "Dimensionless"
IS.Optimization.natural_units(::Type{FixValueParameter}) = "1"
IS.Optimization.output_unit_system(::Type{OnStatusParameter}) = IS.NU
IS.Optimization.quantity_kind(::Type{OnStatusParameter}) = "Dimensionless"
IS.Optimization.natural_units(::Type{OnStatusParameter}) = "1"
IS.Optimization.output_unit_system(::Type{DecrementalCostAtMinParameter}) = IS.NU
IS.Optimization.quantity_kind(::Type{DecrementalCostAtMinParameter}) = "Money"
IS.Optimization.natural_units(::Type{DecrementalCostAtMinParameter}) = "USD"
IS.Optimization.output_unit_system(::Type{FuelCostParameter}) = IS.NU
IS.Optimization.quantity_kind(::Type{FuelCostParameter}) = "Money"
IS.Optimization.natural_units(::Type{FuelCostParameter}) = "USD"
IS.Optimization.output_unit_system(::Type{IncrementalCostAtMinParameter}) = IS.NU
IS.Optimization.quantity_kind(::Type{IncrementalCostAtMinParameter}) = "Money"
IS.Optimization.natural_units(::Type{IncrementalCostAtMinParameter}) = "USD"
IS.Optimization.output_unit_system(::Type{ShutdownCostParameter}) = IS.NU
IS.Optimization.quantity_kind(::Type{ShutdownCostParameter}) = "Money"
IS.Optimization.natural_units(::Type{ShutdownCostParameter}) = "USD"
IS.Optimization.output_unit_system(::Type{DecrementalPiecewiseLinearSlopeParameter}) =
    IS.NU
IS.Optimization.quantity_kind(::Type{DecrementalPiecewiseLinearSlopeParameter}) =
    "CostPerEnergy"
IS.Optimization.natural_units(::Type{DecrementalPiecewiseLinearSlopeParameter}) =
    "USD/MWh"
IS.Optimization.output_unit_system(::Type{IncrementalPiecewiseLinearSlopeParameter}) =
    IS.NU
IS.Optimization.quantity_kind(::Type{IncrementalPiecewiseLinearSlopeParameter}) =
    "CostPerEnergy"
IS.Optimization.natural_units(::Type{IncrementalPiecewiseLinearSlopeParameter}) =
    "USD/MWh"

# Expressions

IS.Optimization.output_unit_system(::Type{ActivePowerBalance}) = IS.SU
IS.Optimization.quantity_kind(::Type{ActivePowerBalance}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{PostContingencyBranchFlow}) = IS.SU
IS.Optimization.quantity_kind(::Type{PostContingencyBranchFlow}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{RawACE}) = IS.SU
IS.Optimization.quantity_kind(::Type{RawACE}) = "ActivePower"
IS.Optimization.output_unit_system(::Type{ReactivePowerBalance}) = IS.SU
IS.Optimization.quantity_kind(::Type{ReactivePowerBalance}) = "ReactivePower"
IS.Optimization.output_unit_system(::Type{DCCurrentBalance}) = IS.SU
IS.Optimization.quantity_kind(::Type{DCCurrentBalance}) = "CurrentFlow"
IS.Optimization.output_unit_system(::Type{CurtailmentCostExpression}) = IS.NU
IS.Optimization.quantity_kind(::Type{CurtailmentCostExpression}) = "Money"
IS.Optimization.natural_units(::Type{CurtailmentCostExpression}) = "USD"
IS.Optimization.output_unit_system(::Type{FixedCostExpression}) = IS.NU
IS.Optimization.quantity_kind(::Type{FixedCostExpression}) = "Money"
IS.Optimization.natural_units(::Type{FixedCostExpression}) = "USD"
IS.Optimization.output_unit_system(::Type{FuelCostExpression}) = IS.NU
IS.Optimization.quantity_kind(::Type{FuelCostExpression}) = "Money"
IS.Optimization.natural_units(::Type{FuelCostExpression}) = "USD"
IS.Optimization.output_unit_system(::Type{ProductionCostExpression}) = IS.NU
IS.Optimization.quantity_kind(::Type{ProductionCostExpression}) = "Money"
IS.Optimization.natural_units(::Type{ProductionCostExpression}) = "USD"
IS.Optimization.output_unit_system(::Type{ShutDownCostExpression}) = IS.NU
IS.Optimization.quantity_kind(::Type{ShutDownCostExpression}) = "Money"
IS.Optimization.natural_units(::Type{ShutDownCostExpression}) = "USD"
IS.Optimization.output_unit_system(::Type{StartUpCostExpression}) = IS.NU
IS.Optimization.quantity_kind(::Type{StartUpCostExpression}) = "Money"
IS.Optimization.natural_units(::Type{StartUpCostExpression}) = "USD"
IS.Optimization.output_unit_system(::Type{VOMCostExpression}) = IS.NU
IS.Optimization.quantity_kind(::Type{VOMCostExpression}) = "Money"
IS.Optimization.natural_units(::Type{VOMCostExpression}) = "USD"
IS.Optimization.output_unit_system(::Type{FuelConsumptionExpression}) = IS.NU
IS.Optimization.quantity_kind(::Type{FuelConsumptionExpression}) = "HeatRate"
IS.Optimization.natural_units(::Type{FuelConsumptionExpression}) = "MMBtu/MWh"

# The approximation variables are internal to the model. Their containers have a second
# axis that is a breakpoint index, not a component, and no consumer reads them.
should_write_resulting_value(::Type{QuadraticVariable}) = false
should_write_resulting_value(::Type{ManualSOS2BinaryVariable}) = false
should_write_resulting_value(::Type{NMDTBinaryVariable}) = false
should_write_resulting_value(::Type{NMDTResidualVariable}) = false
should_write_resulting_value(::Type{NMDTBinaryContinuousProductVariable}) = false
should_write_resulting_value(::Type{NMDTResidualProductVariable}) = false
should_write_resulting_value(::Type{PiecewiseMcCormickBinary}) = false
should_write_resulting_value(::Type{PiecewiseMcCormickDisaggregated}) = false
should_write_resulting_value(::Type{SawtoothAuxVariable}) = false
should_write_resulting_value(::Type{SawtoothBinaryVariable}) = false
should_write_resulting_value(::Type{SawtoothTightenedVariable}) = false
should_write_resulting_value(::Type{BilinearProductVariable}) = false
should_write_resulting_value(::Type{EpigraphVariable}) = false

should_write_resulting_value(::Type{PiecewiseLinearCostVariable}) = false
should_write_resulting_value(::Type{PiecewiseLinearBlockIncrementalOffer}) = false
should_write_resulting_value(::Type{PiecewiseLinearBlockDecrementalOffer}) = false
