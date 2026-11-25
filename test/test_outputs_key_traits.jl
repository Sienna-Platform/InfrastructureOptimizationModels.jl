function _trait_failures(::Type{T}) where {T}
    failures = String[]
    for type in IS.get_all_concrete_subtypes(T)
        if parentmodule(type) !== IOM || !IOM.writes_output_dataset(type)
            continue
        end
        try
            system = ISOPT.output_unit_system(type)
            ISOPT.quantity_kind(type)
            if system == IS.NU
                ISOPT.natural_units(type)
            end
        catch
            push!(failures, string(nameof(type)))
        end
    end
    return failures
end

@testset "IOM key traits" begin
    for T in (IOM.VariableType, IOM.AuxVariableType, IOM.ParameterType, IOM.ExpressionType)
        @test isempty(_trait_failures(T))
    end
    @test IOM.should_write_resulting_value(IOM.QuadraticVariable)
    @test IOM.writes_output_dataset(IOM.QuadraticVariable)
    @test !IOM.should_write_resulting_value(IOM.PiecewiseLinearCostVariable)
    @test IOM.should_write_resulting_value(IOM.FuelConsumptionExpression)
    @test !IOM.writes_output_dataset(IOM.FuelConsumptionExpression)
    @test !IOM.writes_output_dataset(
        IOM.ExpressionKey(IOM.FuelConsumptionExpression, MockComponentType),
    )
end
