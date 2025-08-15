# Measures used
# "classif.prauc", "classif.logloss"


library(data.table)
library(mlr3)
library(mlr3pipelines)
library(mlr3torch)
library(mlr3tuning)
library(mlr3learners)
library(ggplot2)
library(nc)


#####
# Load AUM loss function
#####
source("ProposedAUM.R")

nn_PR_AUM <- torch::nn_module(
  "PR_AUM",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward = Proposed_AUM
)


#####
# Read and setup MNIST task
#####
if (!file.exists("MNIST.csv")) {
  options(timeout = 600)
  install.packages("googledrive")
  library(googledrive)
  file_id <- "1-uFJyJIWZwD5Z58Zzo1XeO8CB45gXO6q"
  drive_download(as_id(file_id), path = "MNIST.csv", overwrite = TRUE)
}

MNIST_dt <- fread("MNIST.csv")
MNIST_dt[, odd := factor(y%%2)]

mtask <- mlr3::TaskClassif$new(
  id = "MNIST",
  backend = MNIST_dt,
  target = "odd"
)
mtask$col_roles$stratum <- "odd"
mtask$col_roles$feature <- grep("^[0-9]+$", names(MNIST_dt), value = TRUE)


#####
# Create neural network learners with different learning rates
#####
measure_list <- msrs(c("classif.logloss", "classif.prauc"))
n.epochs <- 2500

make_torch_learner <- function(id, loss, lr) {
  po_list <- c(
    list(
      mlr3pipelines::po(
        "select",
        selector = mlr3pipelines::selector_type(c("numeric", "integer"))
      ),
      mlr3torch::PipeOpTorchIngressNumeric$new()
    ),
    list(
      mlr3torch::nn("linear", out_features = 1),
      mlr3pipelines::po("nn_head"),
      mlr3pipelines::po("torch_loss", loss),
      mlr3pipelines::po("torch_optimizer", mlr3torch::t_opt("sgd", lr = lr)),
      mlr3pipelines::po("torch_callbacks", mlr3torch::t_clbk("history")),
      mlr3pipelines::po(
        "torch_model_classif",
        batch_size = 100000,
        patience = n.epochs,
        measures_valid = measure_list,
        measures_train = measure_list,
        predict_type = "prob",
        epochs = paradox::to_tune(upper = n.epochs, internal = TRUE)
      )
    )
  )
  graph <- Reduce(mlr3pipelines::concat_graphs, po_list)
  glearner <- mlr3::as_learner(graph)
  mlr3::set_validate(glearner, validate = 0.5)
  mlr3tuning::auto_tuner(
    learner = glearner,
    tuner = mlr3tuning::tnr("internal"),
    resampling = mlr3::rsmp("insample"),
    measure = mlr3::msr("internal_valid_score", minimize = TRUE),
    term_evals = 1,
    id = id,
    store_models = TRUE
  )
}


learning_rates <- c(0.01, 0.05, 0.1)

learner.list <- list()
for (lr in learning_rates) {
  learner.list[[paste0("CrossEntropy_lr_", lr)]] <- make_torch_learner(
    id = paste0("CrossEntropy_lr_", lr),
    loss = torch::nn_bce_with_logits_loss,
    lr = lr
  )
  learner.list[[paste0("PR_AUM_lr_", lr)]] <- make_torch_learner(
    id = paste0("PR_AUM_lr_", lr),
    loss = nn_PR_AUM,
    lr = lr
  )
}


#####
# Benchmark grid
#####
kfoldcv <- rsmp("cv", folds = 3)
bench.grid <- benchmark_grid(
  mtask,
  learner.list,
  kfoldcv
)


#####
# Run benchmark
#####
reg.dir <- "AUM-log&prauc-conv"
cache.RData <- paste0(reg.dir, ".RData")

if (file.exists(cache.RData)) {
  load(cache.RData)
} else {
  if (FALSE) {
    unlink(reg.dir, recursive = TRUE)
    reg = batchtools::makeExperimentRegistry(
      file.dir = reg.dir,
      seed = 1,
      packages = "mlr3verse"
    )
    mlr3batchmark::batchmark(bench.grid, store_models = TRUE, reg = reg)
    job.table <- batchtools::getJobTable(reg = reg)
    chunks <- data.frame(job.table, chunk = 1)
    batchtools::submitJobs(chunks, resources = list(
      walltime = 2 * 60 * 60,
      memory = 6000,
      ncpus = 1,
      ntasks = 1,
      chunks.as.arrayjobs = TRUE
    ), reg = reg)
    batchtools::getStatus(reg=reg)
    jobs.after <- batchtools::getJobTable(reg = reg)
    table(jobs.after$error)
    ids <- jobs.after[is.na(error), job.id]
    bench.result <- mlr3batchmark::reduceResultsBatchmark(ids, reg = reg)
  } else {
    if (require(future)) plan("multisession")
    bench.result <- mlr3::benchmark(bench.grid, store_models = TRUE)
  }
  save(bench.result, file = cache.RData)
}


#####
# Results
#####
score_dt <- bench.result$score()
(score_some <- score_dt[order(classif.ce), .(
  learner_id = factor(learner_id, unique(learner_id)),
  iteration,
  percent_error = 100 * classif.ce
)])


ggplot() +
  geom_point(aes(
    percent_error, learner_id
  ),
  shape = 1,
  data = score_some) +
  scale_x_continuous(
    breaks = seq(0, 100, by = 10),
    limits = c(0, 100)
  )

(score_stats <- data.table::dcast(
  score_some,
  learner_id ~ .,
  list(mean, sd),
  value.var = "percent_error"
))

score_show <- score_stats
ggplot() +
  geom_point(aes(
    percent_error_mean, learner_id
  ),
  shape = 1,
  data = score_stats) +
  geom_segment(aes(
    percent_error_mean + percent_error_sd, learner_id,
    xend = percent_error_mean - percent_error_sd, yend = learner_id
  ),
  data = score_stats) +
  geom_text(aes(
    percent_error_mean, learner_id,
    label = sprintf("%.2f±%.2f", percent_error_mean, percent_error_sd)
  ),
  vjust = -0.5,
  data = score_stats) +
  coord_cartesian(xlim = c(0, 100)) +
  scale_x_continuous(
    "Percent error on test set (mean ± SD over 3 folds in CV)",
    breaks = seq(0, 100, by = 10)
  )

#####
# Curves
#####


torch_learner_ids <- names(learner.list)

(score_torch <- score_dt[
  learner_id %in% torch_learner_ids
][
  , best_epoch := sapply(learner, function(L) {
    unlist(L$tuning_result$internal_tuned_values)
  })
][])

(history_torch <- score_torch[, {
  L <- learner[[1]]
  M <- L$archive$learners(1)[[1]]$model
  M$torch_model_classif$model$callbacks$history
}, by = .(learner_id, iteration)])


(history_long <- nc::capture_melt_single(
  history_torch,
  set = nc::alevels(valid = "validation", train = "subtrain"),
  ".classif.",
  measure = nc::alevels("logloss", "prauc")
))

get_all_folds <- function(DT) {
  DT
}

history_fold <- get_all_folds(history_long)
score_fold <- get_all_folds(score_torch)

score_fold[, learner := paste0(learner_id)]
history_fold[, learner := paste0(learner_id)]

min_fold <- history_fold[
  , .SD[value == min(value)],
  by = .(learner_id, measure, set, iteration)
][, point := "min"]


min_fold <- history_fold[
  , .SD[value == min(value)],
  by = .(learner_id, measure, set, iteration)
][, point := "min"]


ggplot() +
  theme_bw() +
  theme(
    legend.position = "right",
    legend.box = "vertical"
  ) +
  geom_line(aes(
    epoch, value, color = set),
    data = history_fold) +
  geom_point(aes(
    epoch, value, color = set, fill = point),
    shape = 21,
    data = min_fold) +
  scale_fill_manual(
    name = "point",
    values = c(min = "black")
  ) +
  scale_color_manual(
    name = "set",
    values = c("subtrain" = "#17BECF", "validation" = "#FF7F7F")
  ) +
  facet_grid(measure + iteration ~ learner_id, labeller = label_both, scales = "free") +
  scale_x_continuous("epoch") +
  scale_y_continuous("") +
  guides(
    color = guide_legend(title = "set"),
    fill = guide_legend(title = "point")
  )


#####
# Session Info
#####

sessionInfo()

