# FIXME not working and not included in the tests. integration of emulation models in
# POM-IOM split is a work in progress.
@testset "Emulation Model Build" begin
    template = get_thermal_dispatch_template_network()
    c_sys5 = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc";
        add_single_time_series = true,
        force_build = true,
    )

    model = EmulationModel(template, c_sys5; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          IOM.ModelBuildStatus.BUILT
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED

    template = get_thermal_standard_uc_template()
    c_sys5_uc_re = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc_re";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, RenewableDispatch, RenewableFullDispatch)
    model = EmulationModel(template, c_sys5_uc_re; optimizer = HiGHS_optimizer)

    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          IOM.ModelBuildStatus.BUILT
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED
    @test !isempty(collect(readdir(IOM.get_recorder_dir(model))))
end

@testset "Emulation Model initial_conditions test for ThermalGen" begin
    ######## Test with ThermalStandardUnitCommitment ########
    template = get_thermal_standard_uc_template()
    c_sys5_uc_re = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc_re";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, RenewableDispatch, RenewableFullDispatch)
    model = EmulationModel(template, c_sys5_uc_re; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          IOM.ModelBuildStatus.BUILT
    check_duration_on_initial_conditions_values(model, ThermalStandard)
    check_duration_off_initial_conditions_values(model, ThermalStandard)
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED

    ######## Test with ThermalMultiStartUnitCommitment ########
    template = get_thermal_standard_uc_template()
    c_sys5_uc = PSB.build_system(
        PSITestSystems,
        "c_sys5_pglib";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, ThermalMultiStart, ThermalMultiStartUnitCommitment)
    model = EmulationModel(template, c_sys5_uc; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 1, output_dir = mktempdir(; cleanup = true)) ==
          IOM.ModelBuildStatus.BUILT

    check_duration_on_initial_conditions_values(model, ThermalStandard)
    check_duration_off_initial_conditions_values(model, ThermalStandard)
    check_duration_on_initial_conditions_values(model, ThermalMultiStart)
    check_duration_off_initial_conditions_values(model, ThermalMultiStart)
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED

    ######## Test with ThermalStandardUnitCommitment ########
    template = get_thermal_standard_uc_template()
    c_sys5_uc = PSB.build_system(
        PSITestSystems,
        "c_sys5_pglib";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, ThermalMultiStart, ThermalStandardUnitCommitment)
    model = EmulationModel(template, c_sys5_uc; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 1, output_dir = mktempdir(; cleanup = true)) ==
          IOM.ModelBuildStatus.BUILT
    check_duration_on_initial_conditions_values(model, ThermalStandard)
    check_duration_off_initial_conditions_values(model, ThermalStandard)
    check_duration_on_initial_conditions_values(model, ThermalMultiStart)
    check_duration_off_initial_conditions_values(model, ThermalMultiStart)
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED

    ######## Test with ThermalStandardDispatch ########
    template = get_thermal_standard_uc_template()
    c_sys5_uc = PSB.build_system(
        PSITestSystems,
        "c_sys5_pglib";
        add_single_time_series = true,
        force_build = true,
    )
    device_model = DeviceModel(PSY.ThermalStandard, IOM.ThermalStandardDispatch)
    set_device_model!(template, device_model)
    model = EmulationModel(template, c_sys5_uc; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          IOM.ModelBuildStatus.BUILT
end

@testset "Emulation Model initial_conditions test for Hydro" begin
    ######## Test with HydroDispatchRunOfRiver ########
    template = get_thermal_dispatch_template_network()
    c_sys5_hyd = PSB.build_system(
        PSITestSystems,
        "c_sys5_hyd";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, HydroDispatch, HydroDispatchRunOfRiver)
    set_device_model!(template, HydroTurbine, HydroTurbineEnergyDispatch)
    set_device_model!(template, HydroReservoir, HydroEnergyModelReservoir)
    model = EmulationModel(template, c_sys5_hyd; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          IOM.ModelBuildStatus.BUILT
    initial_conditions_data =
        IOM.get_initial_conditions_data(IOM.get_optimization_container(model))
    @test !IOM.has_initial_condition_value(
        initial_conditions_data,
        ActivePowerVariable,
        HydroTurbine,
    )
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED

    ######## Test with HydroCommitmentRunOfRiver ########
    template = get_thermal_dispatch_template_network()
    c_sys5_hyd = PSB.build_system(
        PSITestSystems,
        "c_sys5_hyd";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, HydroDispatch, HydroCommitmentRunOfRiver)
    set_device_model!(template, HydroTurbine, HydroTurbineEnergyCommitment)
    set_device_model!(template, HydroReservoir, HydroEnergyModelReservoir)
    model = EmulationModel(template, c_sys5_hyd; optimizer = HiGHS_optimizer)

    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          IOM.ModelBuildStatus.BUILT
    initial_conditions_data =
        IOM.get_initial_conditions_data(IOM.get_optimization_container(model))
    @test IOM.has_initial_condition_value(
        initial_conditions_data,
        OnVariable,
        HydroTurbine,
    )
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED
end

@testset "Emulation Model Outputs" begin
    template = get_thermal_dispatch_template_network()
    c_sys5 = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc";
        add_single_time_series = true,
        force_build = true,
    )

    model = EmulationModel(template, c_sys5; optimizer = HiGHS_optimizer)
    executions = 10
    @test build!(
        model;
        executions = executions,
        output_dir = mktempdir(; cleanup = true),
    ) ==
          IOM.ModelBuildStatus.BUILT
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED
    outputs = OptimizationProblemOutputs(model)
    @test list_aux_variable_names(outputs) == []
    @test list_aux_variable_keys(outputs) == []
    @test list_variable_names(outputs) == ["ActivePowerVariable__ThermalStandard"]
    @test list_variable_keys(outputs) ==
          [IOM.VariableKey(ActivePowerVariable, ThermalStandard)]
    @test list_dual_names(outputs) == []
    @test list_dual_keys(outputs) == []
    @test list_parameter_names(outputs) == ["ActivePowerTimeSeriesParameter__PowerLoad"]
    @test list_parameter_keys(outputs) ==
          [IOM.ParameterKey(ActivePowerTimeSeriesParameter, PowerLoad)]

    @test read_variable(outputs, "ActivePowerVariable__ThermalStandard") isa DataFrame
    @test read_variable(outputs, ActivePowerVariable, ThermalStandard) isa DataFrame
    @test read_variable(
        outputs,
        IOM.VariableKey(ActivePowerVariable, ThermalStandard),
    ) isa
          DataFrame

    @test read_parameter(outputs, "ActivePowerTimeSeriesParameter__PowerLoad") isa DataFrame
    @test read_parameter(outputs, ActivePowerTimeSeriesParameter, PowerLoad) isa DataFrame
    @test read_parameter(
        outputs,
        IOM.ParameterKey(ActivePowerTimeSeriesParameter, PowerLoad),
    ) isa DataFrame

    @test read_optimizer_stats(model) isa DataFrame
    for n in names(read_optimizer_stats(model))
        stats_values = read_optimizer_stats(model)[!, n]
        if any(ismissing.(stats_values))
            @test ismissing.(stats_values) ==
                  ismissing.(read_optimizer_stats(outputs)[!, n])
        elseif any(isnan.(stats_values))
            @test isnan.(stats_values) == isnan.(read_optimizer_stats(outputs)[!, n])
        else
            @test stats_values == read_optimizer_stats(outputs)[!, n]
        end
    end

    for i in 1:executions
        @test get_objective_value(outputs, i) isa Float64
    end
end

@testset "Run EmulationModel with auto-build" begin
    for serialize in (true, false)
        template = get_thermal_dispatch_template_network()
        c_sys5 = PSB.build_system(
            PSITestSystems,
            "c_sys5_uc";
            add_single_time_series = true,
            force_build = true,
        )

        model = EmulationModel(template, c_sys5; optimizer = HiGHS_optimizer)
        @test_throws ErrorException run!(model, executions = 10)
        @test run!(
            model;
            executions = 10,
            output_dir = mktempdir(; cleanup = true),
            export_optimization_model = serialize,
        ) == IOM.RunStatus.SUCCESSFULLY_FINALIZED
    end
end

@testset "Test serialization/deserialization of EmulationModel outputs" begin
    path = mktempdir(; cleanup = true)
    template = get_thermal_dispatch_template_network()
    c_sys5 = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc";
        add_single_time_series = true,
        force_build = true,
    )

    model = EmulationModel(template, c_sys5; optimizer = HiGHS_optimizer)
    executions = 10
    @test build!(model; executions = executions, output_dir = path) ==
          IOM.ModelBuildStatus.BUILT
    @test run!(model; export_problem_outputs = true) == IOM.RunStatus.SUCCESSFULLY_FINALIZED
    outputs1 = OptimizationProblemOutputs(model)
    var1_a = read_variable(outputs1, ActivePowerVariable, ThermalStandard)
    # Ensure that we can deserialize strings into keys.
    var1_b = read_variable(outputs1, "ActivePowerVariable__ThermalStandard")
    @test var1_a == var1_b

    # Outputs were automatically serialized here.
    outputs2 = OptimizationProblemOutputs(IOM.get_output_dir(model))
    var2 = read_variable(outputs2, ActivePowerVariable, ThermalStandard)
    @test var1_a == var2
    @test get_source_data(outputs2) === nothing
    # Commented out for now, as we no longer automatically serialize the system with results, but this should be added back in the future.
    # load_system(outputs2)
    # @test get_source_data(outputs2) isa PSY.System

    # Serialize to a new directory with the exported function.
    outputs_path = joinpath(path, "outputs")
    serialize_outputs(outputs1, outputs_path)
    @test isfile(joinpath(outputs_path, ISOPT._PROBLEM_OUTPUTS_FILENAME))
    outputs3 = OptimizationProblemOutputs(outputs_path)
    var3 = read_variable(outputs3, ActivePowerVariable, ThermalStandard)
    @test var1_a == var3
    @test get_source_data(outputs3) === nothing
    set_source_data!(outputs3, outputs1.system)
    @test get_source_data(outputs3) !== nothing

    exp_file =
        joinpath(path, "outputs", "variables", "ActivePowerVariable__ThermalStandard.csv")
    var4 = read_dataframe(exp_file)
    # Manually Multiply by the base power var1_a has natural units and export writes directly from the solver
    @test var1_a.value == var4.value .* 100.0
end

@testset "Test serialization of InitialConditionsData" begin
    template = get_thermal_standard_uc_template()
    sys = PSB.build_system(
        PSITestSystems,
        "c_sys5_pglib";
        add_single_time_series = true,
        force_build = true,
    )
    optimizer = HiGHS_optimizer
    set_device_model!(template, ThermalMultiStart, ThermalMultiStartUnitCommitment)
    model = EmulationModel(template, sys; optimizer = HiGHS_optimizer)
    output_dir = mktempdir(; cleanup = true)

    @test build!(model; executions = 1, output_dir = output_dir) ==
          IOM.ModelBuildStatus.BUILT
    ic_file = IOM.get_initial_conditions_file(model)
    test_ic_serialization_outputs(model; ic_file_exists = true, message = "make")
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED

    # Build again, use existing initial conditions.
    IOM.reset!(model)
    @test build!(model; executions = 1, output_dir = output_dir) ==
          IOM.ModelBuildStatus.BUILT
    test_ic_serialization_outputs(model; ic_file_exists = true, message = "make")
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED

    # Build again, use existing initial conditions.
    model = EmulationModel(
        template,
        sys;
        optimizer = optimizer,
        deserialize_initial_conditions = true,
    )
    @test build!(model; executions = 1, output_dir = output_dir) ==
          IOM.ModelBuildStatus.BUILT
    test_ic_serialization_outputs(model; ic_file_exists = true, message = "deserialize")
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED

    # Construct and build again with custom initial conditions file.
    initialization_file = joinpath(output_dir, ic_file * ".old")
    mv(ic_file, initialization_file)
    touch(ic_file)
    model = EmulationModel(
        template,
        sys;
        optimizer = optimizer,
        initialization_file = initialization_file,
        deserialize_initial_conditions = true,
    )
    @test build!(model; executions = 1, output_dir = output_dir) ==
          IOM.ModelBuildStatus.BUILT
    test_ic_serialization_outputs(model; ic_file_exists = true, message = "deserialize")

    # Construct and build again while skipping build of initial conditions.
    model = EmulationModel(template, sys; optimizer = optimizer, initialize_model = false)
    rm(ic_file)
    @test build!(model; executions = 1, output_dir = output_dir) ==
          IOM.ModelBuildStatus.BUILT
    test_ic_serialization_outputs(model; ic_file_exists = false, message = "skip")
    @test run!(model) == IOM.RunStatus.SUCCESSFULLY_FINALIZED
end
