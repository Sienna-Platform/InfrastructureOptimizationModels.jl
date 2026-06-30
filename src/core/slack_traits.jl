# Marker singleton trait selecting whether a model carries slacks. Lifted from the
# former `use_slacks::Bool` field into a model type parameter so slack handling is
# selected by dispatch at compile time rather than a runtime branch.

abstract type SlackUsage end
"Model carries slack variables/constraints."
struct UseSlacks <: SlackUsage end
"Model carries no slacks."
struct NoSlacks <: SlackUsage end

slack_usage(use_slacks::Bool) = use_slacks ? UseSlacks : NoSlacks
