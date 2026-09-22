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
