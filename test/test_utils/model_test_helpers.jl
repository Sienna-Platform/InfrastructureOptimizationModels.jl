period_index_allocs(tm, t) = @allocated IOM.get_period_index(tm, t)

expects_datetime_column(::IOM.AbstractModelStore) = true
expects_datetime_column(::IOM.EmulationModelStore) = false

function set_built_for_recurrent_solves!(container::IOM.OptimizationContainer, val::Bool)
    container.built_for_recurrent_solves = val
    return
end
