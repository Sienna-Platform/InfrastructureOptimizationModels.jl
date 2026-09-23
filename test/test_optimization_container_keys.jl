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

@testset "ComponentPairKey keys" begin
    up_pair = IOM.ComponentPairKey{MockThermalGen, MockReserve{MockUp}}
    down_pair = IOM.ComponentPairKey{MockThermalGen, MockReserve{MockDown}}

    up_key = VariableKey(MockVariable, up_pair, "spin")
    @test IOM.encode_key(up_key) ==
          Symbol("MockVariable__MockThermalGen__MockReserve__MockUp__spin")

    down_key = VariableKey(MockVariable, down_pair, "spin")
    @test up_key != down_key
    @test IOM.encode_key(up_key) != IOM.encode_key(down_key)

    swapped_key = VariableKey(
        MockVariable,
        IOM.ComponentPairKey{MockReserve{MockUp}, MockThermalGen},
        "spin",
    )
    @test swapped_key != up_key
    @test IOM.encode_key(swapped_key) != IOM.encode_key(up_key)

    @test IOM.make_key(VariableKey, MockVariable, up_pair, "spin") == up_key
end

@testset "add_service_variables! keys on (device type, service type)" begin
    time_steps = 1:3
    container = _setup_qa_container(time_steps)
    bus = MockBus("bus", 1, :PV)
    thermal = MockThermalGen("g1", true, bus, (min = 0.0, max = 1.0))
    renewable = MockRenewableGen("g1", true, bus, 1.0)
    contributors = () -> Dict{DataType, Vector{<:IS.InfrastructureSystemsComponent}}(
        MockThermalGen => [thermal],
        MockRenewableGen => [renewable],
    )

    services = Dict(
        MockReserve{MockUp} => MockReserve{MockUp}("spin", 0.5, Any[]),
        MockReserve{MockDown} => MockReserve{MockDown}("spin", 0.25, Any[]),
    )
    for (S, service) in services
        model = ServiceModel(
            S,
            MockReserveFormulation;
            contributing_devices_map = Dict("spin" => contributors()),
        )
        IOM.add_service_variables!(
            container,
            MockVariable,
            [service],
            model,
            MockReserveFormulation,
        )
        @test_throws IS.InvalidValue IOM.add_service_variables!(
            container,
            MockVariable,
            [service],
            model,
            MockReserveFormulation,
        )
    end

    pair_keys = [
        VariableKey(MockVariable, IOM.ComponentPairKey{D, S})
        for D in (MockThermalGen, MockRenewableGen), S in keys(services)
    ]
    @test length(IOM.get_variable_keys(container)) == 4
    @test Set(IOM.get_variable_keys(container)) == Set(pair_keys)
    @test length(unique(IOM.encode_key.(pair_keys))) == 4

    refs = JuMP.VariableRef[]
    for key in pair_keys
        var = IOM.get_variable(container, key)
        @test Set(keys(var.data)) == Set(("spin", "g1", t) for t in time_steps)
        S = IOM.get_component_type(key).parameters[2]
        for t in time_steps
            @test JuMP.upper_bound(var["spin", "g1", t]) == services[S].requirement
            push!(refs, var["spin", "g1", t])
        end
    end
    @test length(unique(refs)) == 4 * length(time_steps)
end
