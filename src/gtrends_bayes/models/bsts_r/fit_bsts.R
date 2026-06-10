# R-side entry points for BSTS. Called from Python via rpy2.

suppressPackageStartupMessages({
  library(bsts)
  library(BoomSpikeSlab)
})

# Source helpers from the directory of THIS file. Robust to the working
# directory rpy2 uses when sourcing.
.this_dir <- function() {
  if (!is.null(sys.frame(1)$ofile)) return(dirname(sys.frame(1)$ofile))
  if (length(sys.frames()) > 0L) {
    fname <- attr(sys.frame(length(sys.frames()))$ofile, "")
    if (!is.null(fname)) return(dirname(fname))
  }
  "."
}
# rpy2 may not populate ofile; safer to require absolute path from Python.
# Python calls source('helpers.R') BEFORE source('fit_bsts.R') so helpers are
# already in scope.

# Registry that keeps fitted models alive so predict() can reach them later
# without round-tripping huge posterior arrays through rpy2.
.gtrends_models <- new.env()

# Fit a BSTS model and stash it under model_id in the registry.
# Returns a list of small/medium-sized posterior summaries that round-trip
# cheaply through rpy2.
fit_bsts <- function(model_id,
                     y,
                     X = NULL,
                     n_seasons = 52L,
                     niter = 3000L,
                     expected_model_size = 5L,
                     seed = 42L,
                     ping = 0L) {
  stopifnot(is.character(model_id), length(model_id) == 1L)
  y <- as.numeric(y)

  ss <- build_state_spec(y, n_seasons = n_seasons)

  set.seed(as.integer(seed))
  if (is.null(X) || (is.data.frame(X) && ncol(X) == 0L)) {
    model <- bsts(y,
                  state.specification = ss,
                  niter = as.integer(niter),
                  ping = as.integer(ping),
                  seed = as.integer(seed))
    has_regression <- FALSE
  } else {
    df <- data.frame(y = y, as.data.frame(X), check.names = FALSE)
    model <- bsts(y ~ .,
                  state.specification = ss,
                  data = df,
                  niter = as.integer(niter),
                  ping = as.integer(ping),
                  expected.model.size = as.integer(expected_model_size),
                  seed = as.integer(seed))
    has_regression <- TRUE
  }
  assign(model_id, model, envir = .gtrends_models)

  one_step_residuals <- tryCatch(
    bsts::bsts.prediction.errors(model)$in.sample,
    error = function(e) NULL
  )

  list(
    has_regression = has_regression,
    niter = as.integer(niter),
    n_obs = length(y),
    state_size = if (!is.null(model$state.contributions))
      dim(model$state.contributions)[2L] else 0L,
    state_contribution_names = dimnames(model$state.contributions)[[2]] %||% character(0),
    sigma_obs = as.numeric(model$sigma.obs),
    coefficients = if (has_regression) model$coefficients else NULL,
    coefficient_names = if (has_regression) colnames(model$coefficients) else character(0),
    inclusion_indicators = if (has_regression) inclusion_indicators(model) else NULL,
    one_step_residuals = one_step_residuals
  )
}

# Posterior forecast `horizon` periods ahead. Returns an (n_draws_kept x horizon)
# matrix of forecast paths. `newdata` must be supplied iff the fitted model has
# a regression component.
predict_bsts <- function(model_id, horizon, newdata = NULL, burn = 0L, quantiles = c(0.025, 0.5, 0.975)) {
  if (!exists(model_id, envir = .gtrends_models)) {
    stop("no fitted BSTS model under id ", model_id)
  }
  model <- get(model_id, envir = .gtrends_models)
  if (is.null(newdata) || (is.data.frame(newdata) && ncol(newdata) == 0L)) {
    pred <- predict(model, horizon = as.integer(horizon),
                    burn = as.integer(burn), quantiles = quantiles)
  } else {
    pred <- predict(model, newdata = as.data.frame(newdata),
                    burn = as.integer(burn), quantiles = quantiles)
  }
  list(
    distribution = pred$distribution,        # (n_draws x horizon)
    mean         = as.numeric(pred$mean),    # (horizon)
    median       = as.numeric(pred$median),  # (horizon)
    quantiles    = pred$interval             # (length(quantiles) x horizon)
  )
}

# Pull the in-sample state-component draws (level / trend / seasonal /
# regression) out of the registry on demand. Returns the raw (niter x n_state x T)
# array; Python wrapper computes posterior bands.
state_contributions <- function(model_id) {
  if (!exists(model_id, envir = .gtrends_models)) {
    stop("no fitted BSTS model under id ", model_id)
  }
  model <- get(model_id, envir = .gtrends_models)
  list(
    contributions = model$state.contributions,
    component_names = dimnames(model$state.contributions)[[2]] %||% character(0)
  )
}

# Free a stored model.
delete_bsts <- function(model_id) {
  if (exists(model_id, envir = .gtrends_models)) {
    rm(list = model_id, envir = .gtrends_models)
  }
  invisible(NULL)
}

# Drop every stored model — useful between walk-forward refits to keep memory in check.
delete_all_bsts <- function() {
  rm(list = ls(envir = .gtrends_models), envir = .gtrends_models)
  invisible(NULL)
}
