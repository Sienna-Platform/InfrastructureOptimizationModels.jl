"""
Common test types used across multiple test files.
Defined in one place to avoid redefinition warnings.

Note: Method definitions for these types (e.g., `sos_status`, `objective_function_multiplier`)
remain in the test files that need them.
"""

# Note: MockContainer is defined in mocks/mock_container.jl with proper fields

#######################################
######## Variable Types ###############
#######################################

struct TestVariableType <: IOM.VariableType end
struct TestVariableType2 <: IOM.VariableType end

ISOPT.output_unit_system(::Type{TestVariableType}) = IS.SU
ISOPT.quantity_kind(::Type{TestVariableType}) = "ActivePower"
ISOPT.output_unit_system(::Type{TestVariableType2}) = IS.SU
ISOPT.quantity_kind(::Type{TestVariableType2}) = "ActivePower"
struct TestCostVariable <: IOM.VariableType end
struct MockVariable <: IOM.VariableType end
struct MockVariable2 <: IOM.VariableType end
struct TestShutDownVariable <: IOM.VariableType end
struct TestStartVariable <: IOM.VariableType end
struct TestOriginalVariable <: IOM.VariableType end
struct TestApproximatedVariable <: IOM.VariableType end

#######################################
######## Sparse Variable Types ########
#######################################

struct TestPWLVariable <: IOM.SparseVariableType end

#######################################
######## Interpolation Variables ######
#######################################

struct TestInterpolationVariable <: IOM.InterpolationVariableType end
struct TestBinaryInterpolationVariable <: IOM.BinaryInterpolationVariableType end

#######################################
######## Expression Types #############
#######################################

struct TestExpressionType <: IOM.ExpressionType end
struct TestCostExpression <: IOM.ExpressionType end
struct MockExpression <: IOM.ExpressionType end
struct MockExpression2 <: IOM.ExpressionType end

#######################################
######## Constraint Types #############
#######################################

struct TestConstraintType <: IOM.ConstraintType end
struct TestCostConstraint <: IOM.ConstraintType end
struct MockConstraint <: IOM.ConstraintType end
struct TestPWLConstraint <: IOM.ConstraintType end

#######################################
######## Auxiliary Variable Types #####
#######################################

struct TestAuxVariableType <: IOM.AuxVariableType end
struct MockAuxVariable <: IOM.AuxVariableType end

#######################################
######## Parameter Types ##############
#######################################

struct TestParameterType <: IOM.ParameterType end
struct TestCostParameter <: IOM.ParameterType end
struct MockParameter <: IOM.ParameterType end

#######################################
######## Initial Condition Types ######
#######################################

struct MockInitialCondition <: IOM.InitialConditionType end

#######################################
######## Formulation Types ############
#######################################

struct TestFormulation <: IOM.AbstractDeviceFormulation end
struct TestPWLFormulation <: IOM.AbstractDeviceFormulation end
struct TestConstantMinGenFormulation <: IOM.AbstractDeviceFormulation end
struct TestCommitmentFormulation <: IOM.AbstractDeviceFormulation end
