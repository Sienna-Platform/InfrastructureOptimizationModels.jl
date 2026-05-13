# The predefined HTML table format recipe was removed in PrettyTables v3.
# This CSS recipe mirrors PT v2 for simple HTML tables.
const tf_html_simple = PrettyTables.HtmlTableFormat(;
    css = """
    table, td, th {
        border-collapse: collapse;
        font-family: sans-serif;
    }

    td, th {
        border-bottom: 0;
        padding: 4px
    }

    tr:nth-child(odd) {
        background: #eee;
    }

    tr:nth-child(even) {
        background: #fff;
    }

    tr.header {
        background: #fff !important;
        font-weight: bold;
    }

    tr.subheader {
        background: #fff !important;
        color: dimgray;
    }

    tr.headerLastRow {
        border-bottom: 2px solid black;
    }

    th.rowNumber, td.rowNumber {
        text-align: right;
    }
    """,
)

function Base.show(io::IO, container::OptimizationContainer)
    show(io, get_jump_model(container))
end

function Base.show(io::IO, ::MIME"text/plain", input::Union{ServiceModel, DeviceModel})
    _show_method(io, input, :auto)
end

function Base.show(io::IO, ::MIME"text/html", input::Union{ServiceModel, DeviceModel})
    _show_method(io, input, :html; stand_alone = false, table_format = tf_html_simple)
end

function _show_method(
    io::IO,
    model::Union{ServiceModel, DeviceModel},
    backend::Symbol;
    kwargs...,
)
    println(io)
    header = ["Device Type", "Formulation", "Slacks"]

    table = Matrix{String}(undef, 1, length(header))
    table[1, 1] = string(get_component_type(model))
    table[1, 2] = string(get_formulation(model))
    table[1, 3] = string(model.use_slacks)

    PrettyTables.pretty_table(
        io,
        table;
        column_labels = header,
        backend = backend,
        title = "Device Model",
        alignment = :l,
        kwargs...,
    )

    if !isempty(model.attributes)
        println(io)
        header = ["Name", "Value"]

        table = Matrix{String}(undef, length(model.attributes), length(header))
        for (ix, (k, v)) in enumerate(model.attributes)
            table[ix, 1] = string(k)
            table[ix, 2] = string(v)
        end

        PrettyTables.pretty_table(
            io,
            table;
            column_labels = header,
            backend = backend,
            title = "Attributes",
            alignment = :l,
            kwargs...,
        )
    end

    if !isempty(model.time_series_names)
        println(io)
        header = ["Parameter Name", "Time Series Name"]

        table = Matrix{String}(undef, length(model.time_series_names), length(header))
        for (ix, (k, v)) in enumerate(model.time_series_names)
            table[ix, 1] = string(k)
            table[ix, 2] = string(v)
        end

        PrettyTables.pretty_table(
            io,
            table;
            column_labels = header,
            backend = backend,
            title = "Time Series Names",
            alignment = :l,
            kwargs...,
        )
    end

    if !isempty(model.duals)
        println(io)

        table = string.(model.duals)

        PrettyTables.pretty_table(
            io,
            table;
            show_column_labels = false,
            backend = backend,
            title = "Duals",
            alignment = :l,
            kwargs...,
        )
    end

    if !isempty(model.feedforwards)
        println(io)
        header = ["Type", "Source", "Affected Values"]
        table = Matrix{String}(undef, length(model.feedforwards), length(header))
        for (ix, v) in enumerate(model.feedforwards)
            table[ix, 1] = string(typeof(v))
            # FIXME this 1-arg version is defined in POM. move it?
            table[ix, 2] = encode_key_as_string(get_optimization_container_key(v))
            table[ix, 3] = first(encode_key_as_string.(get_affected_values(v)))
        end
        PrettyTables.pretty_table(
            io,
            table;
            column_labels = header,
            backend = backend,
            title = "Feedforwards",
            alignment = :l,
            kwargs...,
        )
    else
        println(io)
        print(io, "No FeedForwards Assigned")
    end
end

function Base.show(io::IO, ::MIME"text/plain", input::NetworkModel)
    _show_method(io, input, :auto)
end

function Base.show(io::IO, ::MIME"text/html", input::NetworkModel)
    # The tf_html_simple format was eliminated from PrettyTables and it was added to PowerSystems
    _show_method(io, input, :html; stand_alone = false, table_format = tf_html_simple)
end

function _show_method(io::IO, network_model::NetworkModel, backend::Symbol; kwargs...)
    table = [
        "Network Model" string(get_network_formulation(network_model))
        "Slacks" get_use_slacks(network_model)
        "PTDF" !isnothing(get_PTDF_matrix(network_model))
        "Duals" join(string.(get_duals(network_model)), " ")
    ]

    PrettyTables.pretty_table(
        io,
        table;
        backend = backend,
        column_labels = ["Field", "Value"],
        title = "Network Model",
        alignment = :l,
        kwargs...,
    )
    return
end

function Base.show(io::IO, ::MIME"text/plain", input::OperationModel)
    _show_method(io, input, :auto)
end

function Base.show(io::IO, ::MIME"text/html", input::OperationModel)
    # The tf_html_simple format was eliminated from PrettyTables and it was added to PowerSystems
    _show_method(io, input, :html; stand_alone = false, table_format = tf_html_simple)
end

function _show_method(io::IO, model::OperationModel, backend::Symbol; kwargs...)
    _show_method(io, model.template, backend; kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", input::SimulationModels)
    _show_method(io, input, :auto)
end

function Base.show(io::IO, ::MIME"text/html", input::SimulationModels)
    # The tf_html_simple format was eliminated from PrettyTables and it was added to PowerSystems
    _show_method(io, input, :html; stand_alone = false, table_format = tf_html_simple)
end

function _show_method(io::IO, sim_models::SimulationModels, backend::Symbol; kwargs...)
    println(io)
    header = ["Model Name", "Model Type", "Status", "Output Directory"]

    table = Matrix{Any}(undef, length(sim_models.decision_models), length(header))
    for (ix, model) in enumerate(sim_models.decision_models)
        table[ix, 1] = string(get_name(model))
        table[ix, 2] = IS.strip_module_name(string(get_problem_type(model)))
        table[ix, 3] = string(get_status(model))
        table[ix, 4] = get_output_dir(model)
    end

    PrettyTables.pretty_table(
        io,
        table;
        column_labels = header,
        backend = backend,
        title = "Decision Models",
        alignment = :l,
        kwargs...,
    )

    if !isnothing(sim_models.emulation_model)
        println(io)
        table = Matrix{Any}(undef, 1, length(header))
        table[1, 1] = string(get_name(sim_models.emulation_model))
        table[1, 2] =
            IS.strip_module_name(string(_get_model_type(sim_models.emulation_model)))
        table[1, 3] = string(get_status(sim_models.emulation_model))
        table[1, 4] = get_output_dir(sim_models.emulation_model)

        PrettyTables.pretty_table(
            io,
            table;
            column_labels = header,
            backend = backend,
            title = "Emulator Models",
            alignment = :l,
            kwargs...,
        )
    else
        println(io)
        println(io, "No Emulator Model Specified")
    end
end

function Base.show(io::IO, ::MIME"text/plain", input::SimulationSequence)
    _show_method(io, input, :auto)
end

function Base.show(io::IO, ::MIME"text/html", input::SimulationSequence)
    # The tf_html_simple format was eliminated from PrettyTables and it was added to PowerSystems
    _show_method(io, input, :html; stand_alone = false, table_format = tf_html_simple)
end

function _show_method(io::IO, sequence::SimulationSequence, backend::Symbol; kwargs...)
    println(io)
    table = [
        "Simulation Step Interval" Dates.Hour(get_step_resolution(sequence))
        "Number of Problems" length(sequence.executions_by_model)
    ]

    PrettyTables.pretty_table(
        io,
        table;
        backend = backend,
        show_column_labels = false,
        title = "Simulation Sequence",
        alignment = :l,
        kwargs...,
    )

    println(io)
    header = ["Model Name", "Horizon", "Interval", "Executions Per Step"]

    table = Matrix{Any}(undef, length(sequence.executions_by_model), length(header))
    for (ix, (model, executions)) in enumerate(sequence.executions_by_model)
        table[ix, 1] = string(model)
        table[ix, 2] = Dates.canonicalize(sequence.horizons[model])
        table[ix, 3] = Dates.canonicalize(sequence.intervals[model])
        table[ix, 4] = executions
    end

    PrettyTables.pretty_table(
        io,
        table;
        column_labels = header,
        backend = backend,
        title = "Simulation Problems",
        alignment = :l,
    )

    if !isempty(sequence.feedforwards)
        println(io)
        header = ["Model Name", "Feed Forward Type"]
        table = Matrix{Any}(undef, length(sequence.feedforwards), length(header))
        for (ix, (k, ff)) in enumerate(sequence.feedforwards)
            table[ix, 1] = k
            table[ix, 2] = join(string.(typeof.(ff)), " ")
        end
        PrettyTables.pretty_table(
            io,
            table;
            column_labels = header,
            backend = backend,
            title = "Feedforwards",
            alignment = :l,
            kwargs...,
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", input::Simulation)
    _show_method(io, input, :auto)
end

function Base.show(io::IO, ::MIME"text/html", input::Simulation)
    # The tf_html_simple format was eliminated from PrettyTables and it was added to PowerSystems
    _show_method(io, input, :html; stand_alone = false, table_format = tf_html_simple)
end

function _get_initial_time_for_show(sim::Simulation)
    ini_time = get_initial_time(sim)
    if isnothing(ini_time)
        return "Unset Initial Time"
    else
        return string(ini_time)
    end
end

function _get_build_status_for_show(sim::Simulation)
    internal = sim.internal
    if isnothing(internal)
        return "EMPTY"
    else
        return string(internal.build_status)
    end
end

function _get_run_status_for_show(sim::Simulation)
    internal = sim.internal
    if isnothing(internal)
        return "NOT_READY"
    else
        return string(internal.status)
    end
end

function _show_method(io::IO, sim::Simulation, backend::Symbol; kwargs...)
    table = [
        "Simulation Name" get_name(sim)
        "Build Status" _get_build_status_for_show(sim)
        "Run Status" _get_run_status_for_show(sim)
        "Initial Time" _get_initial_time_for_show(sim)
        "Steps" get_steps(sim)
    ]

    PrettyTables.pretty_table(
        io,
        table;
        backend = backend,
        show_column_labels = false,
        title = "Simulation",
        alignment = :l,
        kwargs...,
    )

    _show_method(io, sim.models, backend; kwargs...)
    _show_method(io, sim.sequence, backend; kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", input::SimulationOutputs)
    _show_method(io, input, :auto)
end

function Base.show(io::IO, ::MIME"text/html", input::SimulationOutputs)
    # The tf_html_simple format was eliminated from PrettyTables and it was added to PowerSystems
    _show_method(io, input, :html; stand_alone = false, table_format = tf_html_simple)
end

function _show_method(io::IO, outputs::SimulationOutputs, backend::Symbol; kwargs...)
    header = ["Problem Name", "Initial Time", "Resolution", "Last Solution Timestamp"]

    table = Matrix{Any}(undef, length(outputs.decision_problem_outputs), length(header))
    for (ix, (key, output)) in enumerate(outputs.decision_problem_outputs)
        table[ix, 1] = key
        table[ix, 2] = first(output.timestamps)
        table[ix, 3] = Dates.canonicalize(output.resolution)
        table[ix, 4] = last(output.timestamps)
    end
    println(io)
    PrettyTables.pretty_table(
        io,
        table;
        column_labels = header,
        backend = backend,
        title = "Decision Problem Outputs",
        alignment = :l,
    )

    println(io)
    table = [
        "Name" outputs.emulation_problem_outputs.problem
        "Resolution" Dates.Minute(outputs.emulation_problem_outputs.resolution)
        "Number of steps" length(outputs.emulation_problem_outputs.timestamps)
    ]
    PrettyTables.pretty_table(
        io,
        table;
        show_column_labels = false,
        backend = backend,
        title = "Emulator Outputs",
        alignment = :l,
        kwargs...,
    )
end

# show methods for OptimizationProblemOutputs, SimulationProblemOutputs,
# ConstraintBounds, VariableBounds, NumericalBounds now live in
# PowerOperationsModels (src/utils/print.jl) along with the concrete types.
