"""
Unit tests for the template deep-copy path and its `share_template_references!` hook.
"""

using Test
using InfrastructureOptimizationModels

# Minimal templates: one shares a field through the hook, one relies on the default.
mutable struct SharingTestTemplate <: IOM.AbstractProblemTemplate
    shared::Vector{Int}
    cloned::Vector{Int}
end
IOM.get_network_model(::SharingTestTemplate) = nothing
function IOM.share_template_references!(
    template_::SharingTestTemplate,
    template::SharingTestTemplate,
)
    template_.shared = template.shared
    return
end

mutable struct DefaultCopyTestTemplate <: IOM.AbstractProblemTemplate
    data::Vector{Int}
end
IOM.get_network_model(::DefaultCopyTestTemplate) = nothing

@testset "Template deep copy" begin
    @testset "share_template_references! runs on the copy" begin
        template = SharingTestTemplate([1], [2])
        template_ = IOM._deepcopy_template(template)
        @test template_ !== template
        @test template_.shared === template.shared
        @test template_.cloned == template.cloned
        @test template_.cloned !== template.cloned
    end

    @testset "default hook shares nothing" begin
        template = DefaultCopyTestTemplate([1])
        template_ = IOM._deepcopy_template(template)
        @test template_.data == template.data
        @test template_.data !== template.data
    end
end

mutable struct EvaluationsTestTemplate <: IOM.AbstractProblemTemplate
    evaluations::IOM.EvaluationContainer
end
IOM.get_evaluations(t::EvaluationsTestTemplate) = t.evaluations
function IOM.set_evaluations!(t::EvaluationsTestTemplate, ec::IOM.EvaluationContainer)
    t.evaluations = ec
    return
end

struct NoEvaluationsTestTemplate <: IOM.AbstractProblemTemplate end

@testset "Template evaluator interface" begin
    err = @test_throws ArgumentError IOM.get_evaluations(NoEvaluationsTestTemplate())
    @test occursin("NoEvaluationsTestTemplate", err.value.msg)
    @test_throws ArgumentError IOM.set_evaluations!(
        NoEvaluationsTestTemplate(),
        IOM.EvaluationContainer(),
    )

    template = EvaluationsTestTemplate(IOM.EvaluationContainer())
    ec = IOM.EvaluationContainer()
    IOM.set_evaluations!(template, ec)
    @test IOM.get_evaluations(template) === ec
end
