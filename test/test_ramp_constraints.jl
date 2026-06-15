@testset "Ramp constraints" begin
    @testset "Initial on/off status for t=1 big-M" begin
        # `_initial_on_status` derives the prior on/off status that gates the t=1 ramp
        # big-M, mirroring the `2 - yprev - ycur` relaxation used for t ≥ 2. Without it
        # the old `1 - ycur` bound made a unit started at t=1 from an off initial state
        # infeasible whenever Pmin > ramp_up·dt.
        # Non-recurrent build: power IC is a Float64, status is exact.
        @test IOM._initial_on_status(0.0) == 0.0          # off
        @test IOM._initial_on_status(0.5) == 1.0          # on
        @test IOM._initial_on_status(IOM.ABSOLUTE_TOLERANCE / 2) == 0.0  # below tolerance -> off
        @test IOM._initial_on_status(2.0 * IOM.ABSOLUTE_TOLERANCE) == 1.0

        # Recurrent build: power IC is a parameter VariableRef whose value is unknown at
        # build time. Conservatively reported as off (0.0) so the t=1 big-M fully relaxes
        # — never reintroducing the start-up infeasibility.
        m = JuMP.Model()
        JuMP.@variable(m, p)
        @test IOM._initial_on_status(p) == 0.0
    end
end
