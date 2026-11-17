period_index_allocs(tm, t) = @allocated IOM.get_period_index(tm, t)

expects_datetime_column(::IOM.AbstractModelStore) = true
expects_datetime_column(::IOM.EmulationModelStore) = false
