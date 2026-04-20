# DCC(1,1)-GARCH(1,1) via full MLE (rmgarch/rugarch).
# Addresses reviewer R3.4: moment-based DCC underestimates the baseline;
# fair comparison needs production DCC.
#
# Args (positional):
#   1. returns_csv    — CSV with Date column + N asset columns
#   2. train_end_idx  — 1-indexed last row to use for fitting (train+val)
#   3. T_oos          — number of OOS days to simulate
#   4. n_paths        — number of Monte Carlo paths
#   5. out_bin        — output path (raw float64 binary, shape (n_paths, T_oos, N))
#   6. seed           — integer seed
#
# Layout: array written as dim (N, T_oos, n_paths) in column-major so that the
# flat byte stream matches Python's row-major reshape to (n_paths, T_oos, N).

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 6)
returns_csv  <- args[1]
train_end    <- as.integer(args[2])
T_oos        <- as.integer(args[3])
n_paths      <- as.integer(args[4])
out_bin      <- args[5]
seed         <- as.integer(args[6])

suppressPackageStartupMessages({
  library(rugarch)
  library(rmgarch)
})

df <- read.csv(returns_csv, check.names = FALSE)
R_full <- as.matrix(df[, -1])
N <- ncol(R_full)
R_fit <- R_full[seq_len(train_end), , drop = FALSE]

# Several assets (pdbc, ftgc, bndx, vtip, meta, govt, cane, schz, flot) have
# large blocks of leading zeros because they started trading later than
# 2011-01-04. Those zero runs push univariate GARCH estimates into
# non-stationary territory (α=1, β=0.5) and crash dccsim. Trim the fitting
# window to the first date where every asset has a non-zero return.
first_nz <- apply(R_fit, 2, function(x) which(x != 0)[1])
start_row <- max(first_nz, na.rm = TRUE)
R_fit <- R_fit[start_row:nrow(R_fit), , drop = FALSE]

# Scale to percent — standard rugarch practice; daily log returns (~0.01)
# are too small for the numerical optimiser.
SCALE <- 100
R_fit <- R_fit * SCALE

cat(sprintf("[dcc_mle.R] Fit data: T=%d (trimmed from row %d), N=%d, OOS=%d, paths=%d (×%d)\n",
            nrow(R_fit), start_row, N, T_oos, n_paths, SCALE))

uspec <- multispec(replicate(N, ugarchspec(
  variance.model     = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model         = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "norm"
)))

mspec <- dccspec(uspec, dccOrder = c(1, 1), distribution = "mvnorm")

t0 <- Sys.time()

# Pre-fit each univariate GARCH with the robust "hybrid" solver and an
# explicit stationarity constraint (α+β<1); rugarch's default solnp often
# fails on assets with long runs of zeros (non-traded early history), and
# without the constraint a few fits land on non-stationary regions which
# then crash dccsim.
prefit <- multifit(uspec, data = R_fit, solver = "hybrid",
                   fit.control = list(stationarity = 1))
conv <- sapply(prefit@fit, function(f) f@fit$convergence)
cat("[dcc_mle.R] univariate prefit: non-converged =",
    sum(conv != 0), "/", length(conv), "\n")

fit <- tryCatch(
  dccfit(mspec, data = R_fit,
         fit.control = list(eval.se = FALSE, scale = FALSE),
         solver = "solnp",
         fit = prefit),
  error = function(e) {
    cat("[dcc_mle.R] dccfit failed:", conditionMessage(e), "\n")
    stop(e)
  }
)
cat(sprintf("[dcc_mle.R] dccfit done in %.1fs\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

t0 <- Sys.time()
sim <- dccsim(fit, n.sim = T_oos, m.sim = n_paths, rseed = seed)
cat(sprintf("[dcc_mle.R] dccsim done in %.1fs\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

# Assemble array with dim (N, T_oos, n_paths). Column-major flatten aligns
# with Python's (n_paths, T_oos, N) row-major reshape.
arr <- array(0, dim = c(N, T_oos, n_paths))
for (p in seq_len(n_paths)) {
  sim_p <- fitted(sim, sim = p)     # (T_oos, N) in percent scale
  arr[, , p] <- t(sim_p) / SCALE    # back to log-return scale
}

con <- file(out_bin, open = "wb")
writeBin(as.double(as.vector(arr)), con, size = 8)
close(con)

cat(sprintf("[dcc_mle.R] Wrote %d doubles to %s (shape=(n_paths=%d, T=%d, N=%d))\n",
            length(arr), out_bin, n_paths, T_oos, N))
