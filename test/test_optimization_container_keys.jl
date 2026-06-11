# mostly a copy-paste from IS tests.
# Test types defined in test_utils/test_types.jl
import InfrastructureOptimizationModels:
    VariableKey,
    ConstraintKey,
    AuxVarKey,
    ExpressionKey,
    ParameterKey,
    InitialConditionKey
import InfrastructureSystems as IS

IOM.convert_output_to_natural_units(::Type{MockVariable2}) = true
IOM.should_write_resulting_value(::Type{MockVariable2}) = false
IOM.convert_output_to_natural_units(::Type{MockExpression2}) = true
IOM.should_write_resulting_value(::Type{MockExpression2}) = false
@testset "Test optimization container keys" begin
    var_key = VariableKey(MockVariable, IS.TestComponent)
    @test IOM.encode_key(var_key) == Symbol("MockVariable__TestComponent")
    constraint_key = ConstraintKey(MockConstraint, IS.TestComponent)
    @test IOM.encode_key(constraint_key) ==
          Symbol("MockConstraint__TestComponent")
    auxvar_key = AuxVarKey(MockAuxVariable, IS.TestComponent)
    @test IOM.encode_key(auxvar_key) == Symbol("MockAuxVariable__TestComponent")
    expression_key = ExpressionKey(MockExpression, IS.TestComponent)
    @test IOM.encode_key(expression_key) ==
          Symbol("MockExpression__TestComponent")
    parameter_key = ParameterKey(MockParameter, IS.TestComponent)
    @test IOM.encode_key(parameter_key) ==
          Symbol("MockParameter__TestComponent")
    ic_key = InitialConditionKey(MockInitialCondition, IS.TestComponent)
    @test IOM.encode_key(ic_key) ==
          Symbol("MockInitialCondition__TestComponent")

    @test_throws ArgumentError ExpressionKey(
        MockExpression,
        IS.InfrastructureSystemsType,
    )

    @test_throws ArgumentError AuxVarKey(
        MockAuxVariable,
        IS.InfrastructureSystemsType,
    )

    # Not tested because it is allowed.
    #@test_throws ArgumentError ConstraintKey(
    #    MockConstraint,
    #    IS.InfrastructureSystemsType,
    #)

    @test_throws ArgumentError VariableKey(
        MockVariable,
        IS.InfrastructureSystemsType,
    )

    @test_throws ArgumentError ParameterKey(
        MockParameter,
        IS.InfrastructureSystemsType,
    )

    @test_throws IS.InvalidValue IOM.check_meta_chars("ZZ__CC")

    # Task 2.11: the key constructor and make_key must validate meta so that a
    # `__` (COMPONENT_NAME_DELIMITER) in meta can't silently corrupt encode/decode.
    @test_throws IS.InvalidValue ConstraintKey(
        MockConstraint,
        IS.TestComponent,
        "bad__meta",
    )
    @test_throws IS.InvalidValue IOM.make_key(
        VariableKey,
        MockVariable,
        IS.TestComponent,
        "bad__meta",
    )

    @test !IOM.convert_output_to_natural_units(var_key)
    @test !IOM.convert_output_to_natural_units(constraint_key)
    @test !IOM.convert_output_to_natural_units(auxvar_key)
    @test !IOM.convert_output_to_natural_units(expression_key)
    @test !IOM.convert_output_to_natural_units(parameter_key)

    @test IOM.should_write_resulting_value(var_key)
    @test IOM.should_write_resulting_value(constraint_key)
    @test IOM.should_write_resulting_value(auxvar_key)
    @test !IOM.should_write_resulting_value(expression_key)
    @test !IOM.should_write_resulting_value(parameter_key)

    var_key2 = VariableKey(MockVariable2, IS.TestComponent)
    @test IOM.convert_output_to_natural_units(var_key2)
    @test !IOM.should_write_resulting_value(var_key2)

    key_strings = IOM.encode_keys_as_strings([var_key, var_key2])
    @test isa(key_strings, Vector{String})

    made_key = IOM.make_key(
        VariableKey,
        MockVariable2,
        IS.TestComponent,
    )
    @test isa(made_key, VariableKey)
end
