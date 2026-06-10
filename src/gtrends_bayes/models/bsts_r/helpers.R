# State-spec helpers for BSTS. Sourced by fit_bsts.R.

`%||%` <- function(a, b) if (!is.null(a)) a else b

# Build the standard state spec used throughout the project:
#   AddLocalLinearTrend  (level + slope, both as random walks)
#   AddSeasonal          (annual cycle when n_seasons > 1)
build_state_spec <- function(y, n_seasons = 52L) {
  ss <- list()
  ss <- bsts::AddLocalLinearTrend(ss, y)
  if (!is.null(n_seasons) && n_seasons > 1) {
    ss <- bsts::AddSeasonal(ss, y, nseasons = as.integer(n_seasons))
  }
  ss
}

# Extract the BoomSpikeSlab inclusion-indicator matrix from a fitted bsts model.
# Returns a (niter x p) matrix where each entry is 1 if predictor j was included
# in MCMC iteration i, else 0. Used by Python to compute posterior inclusion
# probabilities downstream.
inclusion_indicators <- function(model) {
  if (is.null(model$coefficients)) return(NULL)
  (model$coefficients != 0) * 1L
}

# Discard the first `burn` MCMC iterations from every posterior array. Mirrors
# the Python wrapper's burn handling (R's bsts does not auto-discard).
drop_burn <- function(arr, burn) {
  if (is.null(arr) || burn <= 0L) return(arr)
  if (is.null(dim(arr))) {
    return(arr[(burn + 1L):length(arr)])
  }
  d <- dim(arr)
  if (length(d) == 2L) {
    return(arr[(burn + 1L):d[1L], , drop = FALSE])
  }
  if (length(d) == 3L) {
    return(arr[(burn + 1L):d[1L], , , drop = FALSE])
  }
  arr
}
