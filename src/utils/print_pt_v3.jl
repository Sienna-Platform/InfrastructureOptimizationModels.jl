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

function Base.show(io::IO, ::MIME"text/plain", input::AbstractOptimizationModel)
    _show_method(io, input, :auto)
end

function Base.show(io::IO, ::MIME"text/html", input::AbstractOptimizationModel)
    # The tf_html_simple format was eliminated from PrettyTables and it was added to PowerSystems
    _show_method(io, input, :html; stand_alone = false, table_format = tf_html_simple)
end

function _show_method(io::IO, model::AbstractOptimizationModel, backend::Symbol; kwargs...)
    _show_method(io, model.template, backend; kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", input::OptimizationProblemOutputs)
    _show_method(io, input, :auto)
end

function Base.show(io::IO, ::MIME"text/html", input::OptimizationProblemOutputs)
    # The tf_html_simple format was eliminated from PrettyTables and it was added to PowerSystems
    _show_method(io, input, :html; stand_alone = false, table_format = tf_html_simple)
end

function _show_method(
    io::IO,
    outputs::T,
    backend::Symbol;
    kwargs...,
) where {T <: OptimizationProblemOutputs}
    timestamps = get_timestamps(outputs)

    if backend == :html
        println(io, "<p> Start: $(first(timestamps))</p>")
        println(io, "<p> End: $(last(timestamps))</p>")
        println(
            io,
            "<p> Resolution: $(Dates.Minute(get_resolution(outputs)))</p>",
        )
    else
        println(io, "Start: $(first(timestamps))")
        println(io, "End: $(last(timestamps))")
        println(io, "Resolution: $(Dates.Minute(get_resolution(outputs)))")
    end

    values = Dict{String, Vector{String}}(
        "Variables" => list_variable_names(outputs),
        "Auxiliary variables" => list_aux_variable_names(outputs),
        "Duals" => list_dual_names(outputs),
        "Expressions" => list_expression_names(outputs),
        "Parameters" => list_parameter_names(outputs),
    )

    if hasfield(T, :problem)
        name = outputs.problem
    else
        name = "InfrastructureOptimizationModels"
    end

    for (k, val) in values
        if !isempty(val)
            println(io)
            PrettyTables.pretty_table(
                io,
                val;
                show_column_labels = false,
                backend = backend,
                title = "$name Problem $k Outputs",
                alignment = :l,
                kwargs...,
            )
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", bounds::ConstraintBounds)
    println(io, "ConstraintBounds:")
    println(io, "Constraint Coefficient")
    show(io, MIME"text/plain"(), bounds.coefficient)
    println(io, "Constraint RHS")
    show(io, MIME"text/plain"(), bounds.rhs)
end

function Base.show(io::IO, ::MIME"text/plain", bounds::VariableBounds)
    println(io, "VariableBounds:")
    show(io, MIME"text/plain"(), bounds.bounds)
end

function Base.show(io::IO, ::MIME"text/plain", bounds::NumericalBounds)
    println(io, rpad("  Minimum", 20), "Maximum")
    println(io, rpad("  $(bounds.min)", 20), "$(bounds.max)")
end
