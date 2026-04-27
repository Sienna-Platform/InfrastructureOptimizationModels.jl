"""
Unit tests for piecewise linear (PWL) approximation methods.
Tests the mathematical correctness of breakpoint generation.
Test types defined in test_utils/test_types.jl.
"""

# Define sos_status for mock components - return NO_VARIABLE (no commitment tracking)
IOM._sos_status(::Type{MockThermalGen}, ::Type{TestPWLFormulation}) =
    IOM.SOSStatusVariable.NO_VARIABLE

# Define required methods for test types (only non-default values)
IOM.get_variable_binary(
    ::Type{TestBinaryInterpolationVariable},
    ::Type{MockThermalGen},
    ::Type{<:IOM.AbstractDeviceFormulation},
) = true
IOM.get_variable_binary(
    ::Type{<:IOM.VariableType},
    ::Type{MockThermalGen},
    ::Type{<:IOM.AbstractDeviceFormulation},
) = false
IOM.get_variable_upper_bound(
    ::Type{TestInterpolationVariable},
    ::MockThermalGen,
    ::Type{<:IOM.AbstractDeviceFormulation},
) = 1.0
IOM.get_variable_lower_bound(
    ::Type{TestInterpolationVariable},
    ::MockThermalGen,
    ::Type{<:IOM.AbstractDeviceFormulation},
) = 0.0

#==============================================================================#
# Helper functions
#==============================================================================#

function make_mock_thermal_pwl(
    name::String;
    min_power = 10.0,
    max_power = 100.0,
    fuel_cost = 0.0,
)
    bus = MockBus("bus1", 1, :PV)
    op_cost = MockOperationCost(0.0, false, fuel_cost)
    return MockThermalGen(
        name,
        true,
        bus,
        (min = min_power, max = max_power),
        100.0,
        op_cost,
    )
end

function setup_pwl_test_container(time_steps::UnitRange{Int})
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
    )
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, time_steps)
    return container
end

"""
Test breakpoint generation and return the breakpoints for further assertions.
Checks common invariants: correct count, domain boundaries.
"""
function test_breakpoints(f, min_val, max_val, num_segments)
    x_bkpts, y_bkpts =
        IOM._get_breakpoints_for_pwl_function(min_val, max_val, f; num_segments)
    expected_count = num_segments + 1

    @test length(x_bkpts) == expected_count
    @test length(y_bkpts) == expected_count
    @test x_bkpts[1] ≈ min_val
    @test x_bkpts[end] ≈ max_val

    return x_bkpts, y_bkpts
end

"""
Set up container, add PWL interpolation variables, and return relevant objects.
"""
function setup_and_add_pwl_variables(var_type, devices, time_steps, num_segments)
    container = setup_pwl_test_container(time_steps)
    model = IOM.DeviceModel(MockThermalGen, TestPWLFormulation)

    IOM.add_sparse_pwl_interpolation_variables!(
        container,
        var_type,
        devices,
        model,
        num_segments,
    )

    var_container = IOM.get_variable(container, var_type, MockThermalGen)
    jump_model = IOM.get_jump_model(container)

    return (; container, var_container, jump_model)
end

"""
Verify that all expected PWL variables exist and have correct properties.
"""
function verify_pwl_variables(var_container, jump_model, device_names, segment_range,
    time_steps;
    expect_binary, expected_lb = nothing, expected_ub = nothing)
    for name in device_names
        for i in segment_range
            for t in time_steps
                @test haskey(var_container, (name, i, t))
                var = var_container[(name, i, t)]
                @test var isa JuMP.VariableRef
                @test JuMP.is_valid(jump_model, var)
                @test JuMP.is_binary(var) == expect_binary

                if expected_lb !== nothing
                    @test JuMP.lower_bound(var) == expected_lb
                end
                if expected_ub !== nothing
                    @test JuMP.upper_bound(var) == expected_ub
                end
            end
        end
    end
end

"""
Set up a complete PWL constraint test case.

Returns a named tuple with all components needed for testing:
container, x_var, y_var, δ_var, z_var, jump_model, x_bkpts, y_bkpts, devices
"""
function setup_pwl_interpolation_test(;
    device_names::Vector{String} = ["dev1"],
    time_steps::UnitRange{Int} = 1:1,
    num_segments::Int = 4,
    f::Function = x -> x,
    domain::Tuple{Float64, Float64} = (0.0, 10.0),
)
    # Create mock devices
    devices = [make_mock_thermal_pwl(name) for name in device_names]

    # Setup container with all required variables
    container = setup_pwl_test_container(time_steps)
    formulation = TestPWLFormulation
    model = IOM.DeviceModel(MockThermalGen, TestPWLFormulation)

    IOM.add_variables!(container, TestOriginalVariable, devices, formulation)
    IOM.add_variables!(container, TestApproximatedVariable, devices, formulation)
    IOM.add_sparse_pwl_interpolation_variables!(
        container, TestInterpolationVariable, devices, model, num_segments,
    )
    IOM.add_sparse_pwl_interpolation_variables!(
        container, TestBinaryInterpolationVariable, devices, model, num_segments,
    )

    # Generate breakpoints
    x_bkpts, y_bkpts = IOM._get_breakpoints_for_pwl_function(
        domain[1], domain[2], f; num_segments,
    )

    dic_var_bkpts = Dict(name => x_bkpts for name in device_names)
    dic_function_bkpts = Dict(name => y_bkpts for name in device_names)

    # Add constraints
    wrapped_devices = IS.FlattenIteratorWrapper(MockThermalGen, (devices,))

    IOM._add_generic_incremental_interpolation_constraint!(
        container,
        TestOriginalVariable,
        TestApproximatedVariable,
        TestInterpolationVariable,
        TestBinaryInterpolationVariable,
        TestPWLConstraint,
        wrapped_devices,
        dic_var_bkpts,
        dic_function_bkpts,
    )

    return (;
        container,
        x_var = IOM.get_variable(container, TestOriginalVariable, MockThermalGen),
        y_var = IOM.get_variable(container, TestApproximatedVariable, MockThermalGen),
        δ_var = IOM.get_variable(container, TestInterpolationVariable, MockThermalGen),
        z_var = IOM.get_variable(
            container,
            TestBinaryInterpolationVariable,
            MockThermalGen,
        ),
        jump_model = IOM.get_jump_model(container),
        x_bkpts,
        y_bkpts,
        devices,
        num_segments,
        time_steps,
    )
end

"""
Solve the PWL model after fixing δ and z variables. Returns termination status.
"""
function solve_pwl_model!(jump_model, x_var, device_name, t)
    JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
    JuMP.set_silent(jump_model)
    JuMP.@objective(jump_model, Min, x_var[device_name, t])
    JuMP.optimize!(jump_model)
    return JuMP.termination_status(jump_model)
end

#==============================================================================#
# Tests
#==============================================================================#

@testset "PWL Methods" begin
    @testset "_get_breakpoints_for_pwl_function" begin
        @testset "Linear function f(x) = x" begin
            x_bkpts, y_bkpts = test_breakpoints(x -> x, 0.0, 10.0, 5)
            @test y_bkpts ≈ x_bkpts
            @test all(d ≈ 2.0 for d in diff(x_bkpts))
        end

        @testset "Quadratic function f(x) = x^2" begin
            x_bkpts, y_bkpts = test_breakpoints(x -> x^2, 0.0, 4.0, 4)
            @test x_bkpts ≈ [0.0, 1.0, 2.0, 3.0, 4.0]
            @test y_bkpts ≈ [0.0, 1.0, 4.0, 9.0, 16.0]
        end

        @testset "Constant function f(x) = 5" begin
            _, y_bkpts = test_breakpoints(x -> 5.0, 0.0, 10.0, 3)
            @test all(y == 5.0 for y in y_bkpts)
        end

        @testset "Negative domain" begin
            x_bkpts, y_bkpts = test_breakpoints(x -> x^2, -2.0, 2.0, 4)
            @test y_bkpts[1] ≈ y_bkpts[end]  # f(-2) = f(2) = 4
            @test y_bkpts[3] ≈ 0.0           # f(0) = 0
        end

        @testset "Single segment" begin
            x_bkpts, y_bkpts = test_breakpoints(x -> 2x + 1, 0.0, 5.0, 1)
            @test x_bkpts == [0.0, 5.0]
            @test y_bkpts == [1.0, 11.0]
        end

        @testset "Many segments" begin
            x_bkpts, y_bkpts = test_breakpoints(sin, 0.0, Float64(π), 100)
            @test y_bkpts[1] ≈ 0.0 atol = 1e-10
            @test y_bkpts[end] ≈ 0.0 atol = 1e-10
            @test y_bkpts[51] ≈ 1.0 atol = 1e-10  # sin(π/2) = 1
        end

        @testset "Non-integer step sizes" begin
            x_bkpts, _ = test_breakpoints(x -> x, 0.0, 1.0, 3)
            @test x_bkpts ≈ [0.0, 1 / 3, 2 / 3, 1.0]
        end
    end

    @testset "add_sparse_pwl_interpolation_variables!" begin
        @testset "Continuous interpolation variables" begin
            time_steps = 1:3
            num_segments = 4
            devices = [make_mock_thermal_pwl("gen1"), make_mock_thermal_pwl("gen2")]

            setup = setup_and_add_pwl_variables(
                TestInterpolationVariable, devices, time_steps, num_segments,
            )
            @test !isempty(setup.var_container)

            verify_pwl_variables(
                setup.var_container, setup.jump_model,
                get_name.(devices), 1:num_segments, time_steps;
                expect_binary = false, expected_lb = 0.0, expected_ub = 1.0,
            )
        end

        @testset "Binary interpolation variables" begin
            time_steps = 1:2
            num_segments = 4
            devices = [make_mock_thermal_pwl("gen1")]

            setup = setup_and_add_pwl_variables(
                TestBinaryInterpolationVariable, devices, time_steps, num_segments,
            )
            @test !isempty(setup.var_container)

            # Binary variables: num_segments - 1 variables per device
            verify_pwl_variables(
                setup.var_container, setup.jump_model,
                ["gen1"], 1:(num_segments - 1), time_steps;
                expect_binary = true,
            )

            # Should NOT have num_segments-th variable
            @test !haskey(setup.var_container, ("gen1", num_segments, 1))
        end

        @testset "Multiple devices" begin
            time_steps = 1:2
            num_segments = 3
            devices = [make_mock_thermal_pwl("gen$i") for i in 1:3]

            setup = setup_and_add_pwl_variables(
                TestInterpolationVariable, devices, time_steps, num_segments,
            )

            for name in get_name.(devices)
                @test haskey(setup.var_container, (name, 1, 1))
                @test haskey(setup.var_container, (name, num_segments, length(time_steps)))
            end
        end
    end

    @testset "_add_generic_incremental_interpolation_constraint!" begin
        @testset "Constraint structure and count" begin
            test = setup_pwl_interpolation_test(;
                time_steps = 1:2,
                num_segments = 4,
                f = x -> x^2,
            )

            @test IOM.has_container_key(
                test.container,
                TestPWLConstraint,
                MockThermalGen,
                "pwl_variable",
            )
            @test IOM.has_container_key(
                test.container,
                TestPWLConstraint,
                MockThermalGen,
                "pwl_function",
            )

            # 2 main + 2*(num_segments-1) ordering constraints per device per timestep
            expected = (2 + 2 * (test.num_segments - 1)) * 1 * length(test.time_steps)
            @test JuMP.num_constraints(
                test.jump_model;
                count_variable_in_set_constraints = false,
            ) == expected
        end

        @testset "Linear function gives exact PWL" begin
            test = setup_pwl_interpolation_test(;
                num_segments = 3,
                f = x -> x,
                domain = (0.0, 9.0),
            )

            @test test.x_bkpts == test.y_bkpts  # f(x) = x

            # Fix δ = [1.0, 1.0, 0.0] → x = 0 + 1.0*(3) + 1.0*(3) + 0.0*(3) = 6.0
            # z is binary: z[i] >= δ[i+1] and z[i] <= δ[i]
            # z[1] >= δ[2]=1.0 and z[1] <= δ[1]=1.0 → z[1] = 1
            # z[2] >= δ[3]=0.0 and z[2] <= δ[2]=1.0 → z[2] ∈ {0,1}, we choose 0
            JuMP.fix(test.δ_var[("dev1", 1, 1)], 1.0; force = true)
            JuMP.fix(test.δ_var[("dev1", 2, 1)], 1.0; force = true)
            JuMP.fix(test.δ_var[("dev1", 3, 1)], 0.0; force = true)
            JuMP.fix(test.z_var[("dev1", 1, 1)], 1.0; force = true)
            JuMP.fix(test.z_var[("dev1", 2, 1)], 0.0; force = true)

            status = solve_pwl_model!(test.jump_model, test.x_var, "dev1", 1)

            @test status == JuMP.OPTIMAL
            @test JuMP.value(test.x_var["dev1", 1]) ≈ 6.0
            @test JuMP.value(test.y_var["dev1", 1]) ≈ 6.0
        end

        @testset "Quadratic function PWL approximation" begin
            test = setup_pwl_interpolation_test(;
                num_segments = 4,
                f = x -> x^2,
                domain = (0.0, 4.0),
            )

            # Fix δ = [1, 1, 0, 0] → x = 2, y = 4
            JuMP.fix(test.δ_var[("dev1", 1, 1)], 1.0; force = true)
            JuMP.fix(test.δ_var[("dev1", 2, 1)], 1.0; force = true)
            JuMP.fix(test.δ_var[("dev1", 3, 1)], 0.0; force = true)
            JuMP.fix(test.δ_var[("dev1", 4, 1)], 0.0; force = true)
            JuMP.fix(test.z_var[("dev1", 1, 1)], 1.0; force = true)
            JuMP.fix(test.z_var[("dev1", 2, 1)], 0.0; force = true)
            JuMP.fix(test.z_var[("dev1", 3, 1)], 0.0; force = true)

            status = solve_pwl_model!(test.jump_model, test.x_var, "dev1", 1)

            @test status == JuMP.OPTIMAL
            @test JuMP.value(test.x_var["dev1", 1]) ≈ 2.0
            @test JuMP.value(test.y_var["dev1", 1]) ≈ 4.0
        end

        @testset "Multiple devices" begin
            test = setup_pwl_interpolation_test(;
                device_names = ["dev1", "dev2"],
                time_steps = 1:2,
                num_segments = 3,
                f = x -> 2x,
                domain = (0.0, 6.0),
            )

            var_constraints = IOM.get_constraint(
                test.container, TestPWLConstraint, MockThermalGen, "pwl_variable",
            )
            func_constraints = IOM.get_constraint(
                test.container, TestPWLConstraint, MockThermalGen, "pwl_function",
            )

            for name in ["dev1", "dev2"], t in test.time_steps
                @test name in axes(var_constraints, 1) && t in axes(var_constraints, 2)
                @test name in axes(func_constraints, 1) && t in axes(func_constraints, 2)
            end
        end
    end

    @testset "add_variable_cost_to_objective! with FuelCurve{PiecewisePointCurve}" begin
        time_steps = 1:2
        # fuel_cost must match what's in the FuelCurve below
        device = make_mock_thermal_pwl(
            "gen1";
            min_power = 0.0,
            max_power = 100.0,
            fuel_cost = 3.0,
        )
        container = setup_pwl_test_container(time_steps)

        device_name = IOM.get_name(device)
        num_points = 3  # (0, 50, 100) -> 3 breakpoints

        # Add the main variable container for the device
        var_container = IOM.add_variable_container!(
            container,
            TestOriginalVariable,
            MockThermalGen,
            [device_name],
            time_steps,
        )

        # Pre-create PWL variable container with axes: (name, segment_idx, time)
        IOM.add_variable_container!(
            container,
            IOM.PiecewiseLinearCostVariable,
            MockThermalGen,
            [device_name],
            1:num_points,
            time_steps,
        )

        # Populate main variables with actual JuMP variables
        jump_model = IOM.get_jump_model(container)
        for t in time_steps
            var_container[device_name, t] = JuMP.@variable(
                jump_model,
                base_name = "TestPower_$(device_name)_$(t)",
            )
        end

        # FuelCurve with piecewise linear fuel consumption
        # Points: (0, 0), (50, 300), (100, 700) MMBTU
        # Fuel cost: 3.0 $/MMBTU
        fuel_curve = IS.FuelCurve(
            IS.PiecewisePointCurve([(0.0, 0.0), (50.0, 300.0), (100.0, 700.0)]),
            IS.NaturalUnit(),
            3.0,  # $/MMBTU
        )

        IOM.add_variable_cost_to_objective!(
            container,
            TestOriginalVariable,
            device,
            fuel_curve,
            TestPWLFormulation,
        )

        # Verify PWL variables were created
        pwl_vars =
            IOM.get_variable(container, IOM.PiecewiseLinearCostVariable, MockThermalGen)
        for t in time_steps
            for i in 1:num_points
                @test pwl_vars[device_name, i, t] isa JuMP.VariableRef
            end
        end

        # Verify PWL constraints were created
        pwl_constraints = IOM.get_constraint(
            container, IOM.PiecewiseLinearCostConstraint, MockThermalGen,
        )
        for t in time_steps
            @test pwl_constraints[device_name, t] isa JuMP.ConstraintRef
        end
    end
end
