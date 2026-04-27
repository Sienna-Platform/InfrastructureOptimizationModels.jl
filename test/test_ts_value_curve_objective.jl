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

# Helper to create a ForecastKey with sensible defaults
function _make_forecast_key(name::String)
    return IS.ForecastKey(;
        time_series_type = IS.Deterministic,
        name = name,
        initial_timestamp = Dates.DateTime("2020-01-01"),
        resolution = Dates.Hour(1),
        horizon = Dates.Hour(24),
        interval = Dates.Hour(24),
        count = 1,
        features = Dict{String, Any}(),
    )
end

# Helper to create a CostCurve{TimeSeriesPiecewiseIncrementalCurve}
function _make_ts_incremental_cost_curve(;
    power_units::IS.AbstractUnitSystem = IS.NaturalUnit(),
)
    key = _make_forecast_key("test_forecast")
    ii_key = _make_forecast_key("initial_input")
    iaz_key = _make_forecast_key("input_at_zero")
    vc = IS.TimeSeriesPiecewiseIncrementalCurve(key, ii_key, iaz_key)
    return IS.CostCurve(vc, power_units)
end

@testset "TimeSeriesValueCurve Objective Functions" begin
    @testset "TS curve type construction and is_time_series_backed" begin
        key = _make_forecast_key("test_forecast")
        ii_key = _make_forecast_key("initial_input")
        iaz_key = _make_forecast_key("input_at_zero")

        # Construct TimeSeriesPiecewiseIncrementalCurve
        ts_pic = IS.TimeSeriesPiecewiseIncrementalCurve(key, ii_key, iaz_key)
        @test IS.is_time_series_backed(ts_pic)

        # Wrap in CostCurve
        cc = IS.CostCurve(ts_pic)
        @test IS.is_time_series_backed(cc)

        # Verify the type parameter
        @test cc isa IS.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve}
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
                IS.DeviceBaseUnit(),
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
    end
end
