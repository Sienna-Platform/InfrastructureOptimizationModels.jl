# Shared infrastructure for quadratic and bilinear approximations.
#
# Each method ships a scalar `build_<method>` function (pure-JuMP, no IOM
# dependencies) and an `add_<method>_approx!` IOM adapter that allocates
# containers with known `(name, t, ...)` axes, loops over (name, t), calls
# the scalar build per cell, and slots refs into the container.

# --- Abstract config supertypes ---

"Abstract supertype for quadratic approximation method configurations."
abstract type QuadraticApproxConfig end

"Abstract supertype for bilinear approximation method configurations."
abstract type BilinearApproxConfig end

# --- Shared expression / variable key types ---

"Expression container for the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1]."
struct NormedVariableExpression <: ExpressionType end

"Expression container for quadratic (x²) approximation results."
struct QuadraticExpression <: ExpressionType end

"Expression container for bilinear product (x·y) approximation results."
struct BilinearProductExpression <: ExpressionType end

"Variable container for bilinear product (x·y) approximation results."
struct BilinearProductVariable <: VariableType end

"Expression container for sums of two variables, x + y."
struct VariableSumExpression <: ExpressionType end

"Expression container for differences of two variables, x − y."
struct VariableDifferenceExpression <: ExpressionType end
