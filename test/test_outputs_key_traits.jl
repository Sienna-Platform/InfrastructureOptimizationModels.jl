function _trait_failures(::Type{T}) where {T}
    failures = String[]
    for type in IS.get_all_concrete_subtypes(T)
        if parentmodule(type) !== IOM || !IOM.should_write_resulting_value(type)
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
    @test !IOM.should_write_resulting_value(IOM.QuadraticVariable)
end
