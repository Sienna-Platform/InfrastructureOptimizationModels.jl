"""
Tests for CostCurve{TimeSeriesPiecewiseIncrementalCurve} objective function dispatch.
Verifies the PSY-free delta formulation path added in value_curve_cost.jl.
"""

# Formulation dispatch: multiplier and sos_status for test types
IOM.objective_function_multiplier(
    ::Type{TestVariableType},
    ::Type{TestDeviceFormulation},
) = 1.0
IOM._sos_status(::Type, ::Type{TestDeviceFormulation}) = IOM.SOSStatusVariable.NO_VARIABLE

# Keys are store-minted in production; these tests never touch a store (the delta path
# reads from pre-populated parameter containers), so fabricate typed keys directly. The
# ids are arbitrary but distinct.
_make_pwl_forecast_key(id::Int) =
    IS.TimeSeriesKey{IS.Deterministic{IS.PiecewiseStepData}}(id)
_make_scalar_forecast_key(id::Int) = IS.TimeSeriesKey{IS.Deterministic{Float64}}(id)

# Helper to create a CostCurve{TimeSeriesPiecewiseIncrementalCurve}
function _make_ts_incremental_cost_curve(;
    power_units::IS.AbstractUnitSystem = IS.NaturalUnit(),
)
    key = _make_pwl_forecast_key(1)
    ii_key = _make_scalar_forecast_key(2)
    iaz_key = _make_scalar_forecast_key(3)
    vc = IS.TimeSeriesPiecewiseIncrementalCurve(key, ii_key, iaz_key)
    return IS.CostCurve(vc, power_units)
end

@testset "TimeSeriesValueCurve Objective Functions" begin
    @testset "TS curve type construction and is_time_series_backed" begin
        key = _make_pwl_forecast_key(1)
        ii_key = _make_scalar_forecast_key(2)
        iaz_key = _make_scalar_forecast_key(3)

        # Construct TimeSeriesPiecewiseIncrementalCurve
        ts_pic = IS.TimeSeriesPiecewiseIncrementalCurve(key, ii_key, iaz_key)
        @test IS.is_time_series_backed(ts_pic)

        # Wrap in CostCurve
        cc = IS.CostCurve(ts_pic)
        @test IS.is_time_series_backed(cc)

        # Verify the type parameter (the alias is a UnionAll now that keys carry the
        # stored time series type, so match covariantly)
        @test cc isa IS.CostCurve{<:IS.TimeSeriesPiecewiseIncrementalCurve}
    end

    @testset "is_nontrivial_offer reads the association id" begin
        placeholder = IS.CostCurve(
            IS.TimeSeriesPiecewiseIncrementalCurve(
                _make_pwl_forecast_key(0),
                _make_scalar_forecast_key(0),
                _make_scalar_forecast_key(0),
            ),
        )
        @test !IOM.is_nontrivial_offer(placeholder)
        @test IOM.is_nontrivial_offer(_make_ts_incremental_cost_curve())
    end

    @testset "Delta formulation with static parameters" begin
        time_steps = 1:3
        names = ["gen1"]
        # 3 segments: breakpoints [0, 50, 100, 150], slopes [10.0, 20.0, 30.0]
        breakpoints = [0.0, 50.0, 100.0, 150.0]
        slopes = [10.0, 20.0, 30.0]

        container = make_test_container(time_steps; base_power = 100.0)

        # Add power variable
        for t in time_steps
            add_test_variable!(container, TestVariableType, MockThermalGen, "gen1", t)
        end

        # Add expression container
        add_test_expression!(
            container, IOM.ProductionCostExpression, MockThermalGen, names, time_steps)

        # Populate slope/breakpoint parameters
        slopes_mat = [slopes for _ in 1:length(names), _ in time_steps]
        bp_mat = [breakpoints for _ in 1:length(names), _ in time_steps]
        setup_delta_pwl_parameters!(
            container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

        cost_fn = _make_ts_incremental_cost_curve()
        device = make_mock_thermal("gen1")

        # Call the new dispatch
        IOM.add_variable_cost_to_objective!(
            container,
            TestVariableType,
            device,
            cost_fn,
            TestDeviceFormulation,
        )

        # Verify delta variables were created (PiecewiseLinearBlockIncrementalOffer)
        @test IOM.has_container_key(
            container,
            IOM.PiecewiseLinearBlockIncrementalOffer,
            MockThermalGen,
        )

        # Verify block constraints were created
        @test IOM.has_container_key(
            container,
            IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
            MockThermalGen,
        )

        # Verify cost is in variant objective (time-series-backed = always variant)
        @test count_objective_terms(container; variant = true) > 0

        # With base_power=100, NATURAL_UNITS:
        # breakpoints normalized: bp / base_power = [0, 0.5, 1.0, 1.5]
        # slopes normalized: slope * base_power = [1000, 2000, 3000]
        # dt = 1.0 (hourly), sign = +1.0
        # Expected coefficients for each delta var: slope_normalized * dt
        expected_slopes_normalized = slopes .* 100.0  # [1000, 2000, 3000]
        dt = 1.0

        delta_var_container = IOM.get_variable(
            container, IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen)
        for t in time_steps
            for (k, expected_slope) in enumerate(expected_slopes_normalized)
                delta_var = delta_var_container[("gen1", k, t)]
                obj = IOM.get_objective_expression(container)
                variant = IOM.get_variant_terms(obj)
                coeff = JuMP.coefficient(variant, delta_var)
                @test isapprox(coeff, expected_slope * dt; atol = 1e-10)
            end
        end
    end

    @testset "Time-varying slopes across time steps" begin
        time_steps = 1:2
        names = ["gen1"]
        breakpoints = [0.0, 50.0, 100.0]

        # Different slopes per time step
        slopes_t1 = [10.0, 20.0]
        slopes_t2 = [15.0, 25.0]
        time_varying_slopes = Matrix{Vector{Float64}}(undef, 1, 2)
        time_varying_slopes[1, 1] = slopes_t1
        time_varying_slopes[1, 2] = slopes_t2

        container = make_test_container(time_steps; base_power = 100.0)

        for t in time_steps
            add_test_variable!(container, TestVariableType, MockThermalGen, "gen1", t)
        end
        add_test_expression!(
            container, IOM.ProductionCostExpression, MockThermalGen, names, time_steps)

        bp_mat = [breakpoints for _ in 1:length(names), _ in time_steps]
        setup_delta_pwl_parameters!(
            container, MockThermalGen, names, time_varying_slopes, bp_mat, time_steps)

        cost_fn = _make_ts_incremental_cost_curve()
        device = make_mock_thermal("gen1")

        IOM.add_variable_cost_to_objective!(
            container, TestVariableType, device, cost_fn, TestDeviceFormulation)

        delta_var_container = IOM.get_variable(
            container, IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen)
        obj = IOM.get_objective_expression(container)
        variant = IOM.get_variant_terms(obj)

        # t=1: slopes_t1 * base_power * dt = [1000, 2000]
        # t=2: slopes_t2 * base_power * dt = [1500, 2500]
        for (k, s) in enumerate(slopes_t1)
            coeff = JuMP.coefficient(variant, delta_var_container[("gen1", k, 1)])
            @test isapprox(coeff, s * 100.0; atol = 1e-10)
        end
        for (k, s) in enumerate(slopes_t2)
            coeff = JuMP.coefficient(variant, delta_var_container[("gen1", k, 2)])
            @test isapprox(coeff, s * 100.0; atol = 1e-10)
        end
    end

    # POM populates parameter arrays, not IOM: this is really a test of our test utils,
    # `setup_delta_pwl_parameters!` in test_utils/objective_function_helpers.jl
    @testset "Varying tranche counts across devices" begin
        # gen1 has 3 segments, gen2 has 2. Parameter arrays are padded to max (3).
        # Padded slopes = 0, padded breakpoints = last breakpoint (zero-width tranches).
        time_steps = 1:1
        names = ["gen1", "gen2"]
        base_power = 100.0

        slopes_mat = Matrix{Vector{Float64}}(undef, 2, 1)
        slopes_mat[1, 1] = [10.0, 20.0, 30.0]  # gen1: 3 segments
        slopes_mat[2, 1] = [5.0, 15.0]           # gen2: 2 segments

        bp_mat = Matrix{Vector{Float64}}(undef, 2, 1)
        bp_mat[1, 1] = [0.0, 50.0, 100.0, 150.0]  # gen1: 4 breakpoints
        bp_mat[2, 1] = [0.0, 40.0, 80.0]            # gen2: 3 breakpoints

        container = make_test_container(time_steps; base_power = base_power)
        # Create variable container with both names upfront (add_test_variable!
        # creates the axis from the first name only)
        IOM.add_variable_container!(
            container, TestVariableType, MockThermalGen, names, time_steps)
        jump_model = IOM.get_jump_model(container)
        var_container = IOM.get_variable(container, TestVariableType, MockThermalGen)
        for name in names, t in time_steps
            var_container[name, t] =
                JuMP.@variable(jump_model, base_name = "TestVar_$(name)_$(t)")
        end
        add_test_expression!(
            container, IOM.ProductionCostExpression, MockThermalGen, names, time_steps)

        setup_delta_pwl_parameters!(
            container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

        cost_fn = _make_ts_incremental_cost_curve()

        # Run objective for both devices
        for name in names
            device = make_mock_thermal(name)
            IOM.add_variable_cost_to_objective!(
                container, TestVariableType, device, cost_fn, TestDeviceFormulation)
        end

        delta_var_container = IOM.get_variable(
            container, IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen)
        obj = IOM.get_objective_expression(container)
        variant = IOM.get_variant_terms(obj)

        # gen1: 3 real segments, coefficients = slopes * base_power
        for (k, s) in enumerate([10.0, 20.0, 30.0])
            coeff = JuMP.coefficient(variant, delta_var_container[("gen1", k, 1)])
            @test isapprox(coeff, s * base_power; atol = 1e-10)
        end

        # gen2: 2 real segments with correct costs, 3rd segment has zero cost (padded)
        for (k, s) in enumerate([5.0, 15.0])
            coeff = JuMP.coefficient(variant, delta_var_container[("gen2", k, 1)])
            @test isapprox(coeff, s * base_power; atol = 1e-10)
        end
        # Padded segment: slope = 0 → zero objective coefficient
        padded_coeff = JuMP.coefficient(variant, delta_var_container[("gen2", 3, 1)])
        @test isapprox(padded_coeff, 0.0; atol = 1e-10)
    end

    @testset "Resolution scaling (15-min)" begin
        time_steps = 1:2
        names = ["gen1"]
        breakpoints = [0.0, 50.0, 100.0]
        slopes = [10.0, 20.0]

        container = make_test_container(
            time_steps; base_power = 100.0, resolution = Dates.Minute(15))

        for t in time_steps
            add_test_variable!(container, TestVariableType, MockThermalGen, "gen1", t)
        end
        add_test_expression!(
            container, IOM.ProductionCostExpression, MockThermalGen, names, time_steps)

        slopes_mat = [slopes for _ in 1:length(names), _ in time_steps]
        bp_mat = [breakpoints for _ in 1:length(names), _ in time_steps]
        setup_delta_pwl_parameters!(
            container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

        cost_fn = _make_ts_incremental_cost_curve()
        device = make_mock_thermal("gen1")

        IOM.add_variable_cost_to_objective!(
            container, TestVariableType, device, cost_fn, TestDeviceFormulation)

        # dt = 15min / 60min = 0.25
        dt = 0.25
        delta_var_container = IOM.get_variable(
            container, IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen)
        obj = IOM.get_objective_expression(container)
        variant = IOM.get_variant_terms(obj)

        for t in time_steps
            for (k, s) in enumerate(slopes)
                coeff = JuMP.coefficient(variant, delta_var_container[("gen1", k, t)])
                @test isapprox(coeff, s * 100.0 * dt; atol = 1e-10)
            end
        end
    end

    @testset "Decremental direction" begin
        time_steps = 1:2
        names = ["gen1"]
        breakpoints = [0.0, 50.0, 100.0]
        slopes = [10.0, 20.0]

        container = make_test_container(time_steps; base_power = 100.0)

        for t in time_steps
            add_test_variable!(container, TestVariableType, MockThermalGen, "gen1", t)
        end
        add_test_expression!(
            container, IOM.ProductionCostExpression, MockThermalGen, names, time_steps)

        slopes_mat = [slopes for _ in 1:length(names), _ in time_steps]
        bp_mat = [breakpoints for _ in 1:length(names), _ in time_steps]
        setup_delta_pwl_parameters!(
            container, MockThermalGen, names, slopes_mat, bp_mat, time_steps;
            dir = IOM.DecrementalOffer())

        cost_fn = _make_ts_incremental_cost_curve()
        device = make_mock_thermal("gen1")

        IOM.add_variable_cost_to_objective!(
            container, TestVariableType, device, cost_fn, TestDeviceFormulation;
            dir = IOM.DecrementalOffer())

        # Verify decremental variable and constraint types used
        @test IOM.has_container_key(
            container,
            IOM.PiecewiseLinearBlockDecrementalOffer,
            MockThermalGen,
        )
        @test IOM.has_container_key(
            container,
            IOM.PiecewiseLinearBlockDecrementalOfferConstraint,
            MockThermalGen,
        )

        # Verify negative sign: OBJECTIVE_FUNCTION_NEGATIVE = -1.0
        dt = 1.0
        delta_var_container = IOM.get_variable(
            container, IOM.PiecewiseLinearBlockDecrementalOffer, MockThermalGen)
        obj = IOM.get_objective_expression(container)
        variant = IOM.get_variant_terms(obj)

        for t in time_steps
            for (k, s) in enumerate(slopes)
                coeff = JuMP.coefficient(variant, delta_var_container[("gen1", k, t)])
                expected = s * 100.0 * dt * IOM.OBJECTIVE_FUNCTION_NEGATIVE
                @test isapprox(coeff, expected; atol = 1e-10)
            end
        end
    end

    @testset "Unit system conversion" begin
        # Use system_base_power=100, device_base_power=50 to make conversions visible.
        # Input data (slopes and breakpoints) is in the given unit system.
        # Expected output is always in system per-unit.
        #
        # NATURAL_UNITS: breakpoints in MW, slopes in $/MW
        #   bp_pu = bp / system_base   slopes_pu = slopes * system_base
        # DEVICE_BASE: breakpoints in device p.u., slopes in $/device_p.u.
        #   ratio = device_base / system_base = 0.5
        #   bp_pu = bp * ratio          slopes_pu = slopes / ratio
        # SYSTEM_BASE: already in system p.u.
        #   bp_pu = bp                  slopes_pu = slopes

        system_base = 100.0
        device_base = 50.0
        time_steps = 1:1
        names = ["gen1"]
        raw_slopes = [10.0, 20.0]
        raw_breakpoints = [0.0, 50.0, 100.0]

        for (unit_system, expected_slope_factor, expected_bp_factor) in [
            (IS.NaturalUnit(), system_base, 1.0 / system_base),
            (
                IS.ComponentBaseUnit(),
                1.0 / (device_base / system_base),
                device_base / system_base,
            ),
            (IS.SystemBaseUnit(), 1.0, 1.0),
        ]
            @testset "$unit_system" begin
                container = make_test_container(time_steps; base_power = system_base)
                for t in time_steps
                    add_test_variable!(
                        container, TestVariableType, MockThermalGen, "gen1", t)
                end
                add_test_expression!(
                    container, IOM.ProductionCostExpression, MockThermalGen,
                    names, time_steps)

                slopes_mat = [raw_slopes for _ in 1:1, _ in time_steps]
                bp_mat = [raw_breakpoints for _ in 1:1, _ in time_steps]
                setup_delta_pwl_parameters!(
                    container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

                cost_fn = _make_ts_incremental_cost_curve(; power_units = unit_system)
                device = make_mock_thermal("gen1"; base_power = device_base)

                IOM.add_variable_cost_to_objective!(
                    container, TestVariableType, device, cost_fn,
                    TestDeviceFormulation)

                # Check objective coefficients reflect the converted slopes
                dt = 1.0
                delta_var_container = IOM.get_variable(
                    container,
                    IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen)
                obj = IOM.get_objective_expression(container)
                variant = IOM.get_variant_terms(obj)

                for (k, s) in enumerate(raw_slopes)
                    coeff = JuMP.coefficient(
                        variant, delta_var_container[("gen1", k, 1)])
                    expected = s * expected_slope_factor * dt
                    @test isapprox(coeff, expected; atol = 1e-10)
                end
            end
        end
    end

    @testset "add_pwl_constraint_delta! accepts mock types" begin
        # Direct call with MockThermalGen to confirm widened type bound works
        time_steps = 1:1
        container = make_test_container(time_steps; base_power = 100.0)

        device = make_mock_thermal("gen1")

        # Add variable container
        add_test_variable!(container, TestVariableType, MockThermalGen, "gen1", 1)

        breakpoints = [0.0, 0.5, 1.0]
        pwl_vars = IOM.add_pwl_variables_delta!(
            container,
            IOM.PiecewiseLinearBlockIncrementalOffer,
            MockThermalGen,
            "gen1",
            1,
            2;
            upper_bound = Inf,
        )

        # This should not throw with the widened type bound
        IOM.add_pwl_constraint_delta!(
            container,
            device,
            TestVariableType,
            TestDeviceFormulation,
            breakpoints,
            pwl_vars,
            1,
            IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
        )

        @test IOM.has_container_key(
            container,
            IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
            MockThermalGen,
        )
    end

    @testset "add_pwl_constraint_delta! with constant min-gen offset" begin
        # Formulation where _include_constant_min_gen_power_in_constraint returns true
        IOM._include_constant_min_gen_power_in_constraint(
            ::Type{MockThermalGen},
            ::Type{TestVariableType},
            ::Type{TestConstantMinGenFormulation},
        ) = true

        time_steps = 1:1
        container = make_test_container(time_steps; base_power = 100.0)
        device = make_mock_thermal("gen1")

        add_test_variable!(container, TestVariableType, MockThermalGen, "gen1", 1)

        breakpoints = [10.0, 50.0, 100.0]
        pwl_vars = IOM.add_pwl_variables_delta!(
            container,
            IOM.PiecewiseLinearBlockIncrementalOffer,
            MockThermalGen,
            "gen1",
            1,
            2;
            upper_bound = Inf,
        )

        IOM.add_pwl_constraint_delta!(
            container,
            device,
            TestVariableType,
            TestConstantMinGenFormulation,
            breakpoints,
            pwl_vars,
            1,
            IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
        )

        @test IOM.has_container_key(
            container,
            IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
            MockThermalGen,
        )
        con_rhs(c) = JuMP.normalized_rhs(
            IOM.get_constraint(
                c, IOM.PiecewiseLinearBlockIncrementalOfferConstraint, MockThermalGen,
            )[
                "gen1",
                1,
            ],
        )
        @test con_rhs(container) == 10.0

        # An OnStatusParameter (e.g. a SemiContinuousFeedforward) scales the offset.
        function build_with_on_status(names, on_status; eltype = Float64)
            c = make_test_container(time_steps; base_power = 100.0)
            add_test_variable!(c, TestVariableType, MockThermalGen, "gen1", 1)
            add_test_parameter!(
                c, IOM.OnStatusParameter, MockThermalGen, names, time_steps,
                fill(on_status, length(names), 1); eltype = eltype,
            )
            vars = IOM.add_pwl_variables_delta!(
                c, IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen, "gen1", 1,
                2;
                upper_bound = Inf,
            )
            IOM.add_pwl_constraint_delta!(
                c, device, TestVariableType, TestConstantMinGenFormulation, breakpoints,
                vars, 1, IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
            )
            return c
        end
        for on_status in (0.0, 1.0)
            @test con_rhs(build_with_on_status(["gen1"], on_status)) == 10.0 * on_status
        end

        # Recurrent builds hold the status as a JuMP parameter: the offset is 10 * status.
        c = build_with_on_status(["gen1"], 1.0; eltype = JuMP.VariableRef)
        on_param =
            IOM.get_parameter_array(c, IOM.OnStatusParameter, MockThermalGen)["gen1", 1]
        con = IOM.get_constraint(
            c, IOM.PiecewiseLinearBlockIncrementalOfferConstraint, MockThermalGen,
        )[
            "gen1",
            1,
        ]
        @test JuMP.normalized_coefficient(con, on_param) == -10.0
        @test con_rhs(c) == 0.0

        # A device outside the parameter's axis (e.g. must-run) keeps the full offset.
        @test con_rhs(build_with_on_status(["gen2"], 0.0)) == 10.0

        # update_pwl_offset! moves the row to a new first breakpoint the way the build sets it.
        function update_to_30!(c)
            row = IOM.get_constraint(
                c, IOM.PiecewiseLinearBlockIncrementalOfferConstraint, MockThermalGen,
            )[
                "gen1",
                1,
            ]
            IOM.update_pwl_offset!(
                c, row, device, TestVariableType, TestConstantMinGenFormulation, 1, 30.0,
            )
            return row
        end
        update_to_30!(container)
        @test con_rhs(container) == 30.0
        for on_status in (0.0, 1.0)
            c = build_with_on_status(["gen1"], on_status)
            update_to_30!(c)
            @test con_rhs(c) == 30.0 * on_status
        end
        c = build_with_on_status(["gen1"], 1.0; eltype = JuMP.VariableRef)
        on_param =
            IOM.get_parameter_array(c, IOM.OnStatusParameter, MockThermalGen)["gen1", 1]
        row = update_to_30!(c)
        @test JuMP.normalized_coefficient(row, on_param) == -30.0
        @test con_rhs(c) == 0.0
        c = build_with_on_status(["gen2"], 0.0)
        update_to_30!(c)
        @test con_rhs(c) == 30.0
    end

    @testset "add_pwl_constraint_delta! with OnVariable min-gen offset" begin
        # Formulation where _include_min_gen_power_in_constraint returns true
        IOM._include_min_gen_power_in_constraint(
            ::Type{MockThermalGen},
            ::Type{TestVariableType},
            ::Type{TestCommitmentFormulation},
        ) = true

        time_steps = 1:1
        container = make_test_container(time_steps; base_power = 100.0)
        device = make_mock_thermal("gen1")

        add_test_variable!(container, TestVariableType, MockThermalGen, "gen1", 1)

        # Add OnVariable for the commitment path
        on_var_container = IOM.add_variable_container!(
            container, IOM.OnVariable, MockThermalGen, ["gen1"], time_steps)
        jump_model = IOM.get_jump_model(container)
        on_var_container["gen1", 1] = JuMP.@variable(
            jump_model, base_name = "On_gen1_1", binary = true)

        breakpoints = [10.0, 50.0, 100.0]
        pwl_vars = IOM.add_pwl_variables_delta!(
            container,
            IOM.PiecewiseLinearBlockIncrementalOffer,
            MockThermalGen,
            "gen1",
            1,
            2;
            upper_bound = Inf,
        )

        IOM.add_pwl_constraint_delta!(
            container,
            device,
            TestVariableType,
            TestCommitmentFormulation,
            breakpoints,
            pwl_vars,
            1,
            IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
        )

        @test IOM.has_container_key(
            container,
            IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
            MockThermalGen,
        )
        row = IOM.get_constraint(
            container, IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
            MockThermalGen,
        )[
            "gen1",
            1,
        ]
        on_var = on_var_container["gen1", 1]
        @test JuMP.normalized_coefficient(row, on_var) == -10.0

        IOM.update_pwl_offset!(
            container, row, device, TestVariableType, TestCommitmentFormulation, 1, 30.0,
        )
        @test JuMP.normalized_coefficient(row, on_var) == -30.0
        @test JuMP.normalized_rhs(row) == 0.0
    end

    @testset "add_pwl_constraint_delta! measures PowerAboveMinimumVariable from P_min" begin
        # Compact form: p̂ = p - P_min·on, so the row is p̂ - Σδ - (P_0 - P_min)·on == 0.
        IOM._include_min_gen_power_in_constraint(
            ::Type{MockThermalGen},
            ::Type{IOM.PowerAboveMinimumVariable},
            ::Type{TestCommitmentFormulation},
        ) = true
        container = make_test_container(1:1; base_power = 100.0)
        device = make_mock_thermal("gen1"; limits = (min = 20.0, max = 100.0))
        add_test_variable!(
            container, IOM.PowerAboveMinimumVariable, MockThermalGen, "gen1", 1)
        on_vars = IOM.add_variable_container!(
            container, IOM.OnVariable, MockThermalGen, ["gen1"], 1:1)
        on_var =
            on_vars["gen1", 1] = JuMP.@variable(
                IOM.get_jump_model(container), base_name = "On_gen1_1", binary = true)
        pwl_vars = IOM.add_pwl_variables_delta!(
            container, IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen, "gen1",
            1, 2;
            upper_bound = Inf,
        )
        IOM.add_pwl_constraint_delta!(
            container, device, IOM.PowerAboveMinimumVariable, TestCommitmentFormulation,
            [10.0, 50.0, 100.0], pwl_vars, 1,
            IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
        )
        row = IOM.get_constraint(
            container, IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
            MockThermalGen,
        )[
            "gen1",
            1,
        ]
        @test JuMP.normalized_coefficient(row, on_var) == 10.0
        @test JuMP.normalized_rhs(row) == 0.0

        IOM.update_pwl_offset!(
            container, row, device, IOM.PowerAboveMinimumVariable,
            TestCommitmentFormulation, 1, 30.0,
        )
        @test JuMP.normalized_coefficient(row, on_var) == -10.0
    end
end
