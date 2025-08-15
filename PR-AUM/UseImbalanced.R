# Measures used
# "classif.prauc", "classif.acc"


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

nn_AUM_loss <- torch::nn_module(
  "nn_AUM_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward = Proposed_AUM
)


#####
# Read and setup tasks from unbalanced data
#####
unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/*csv")
task.list <- list()

for(unb.csv in unb.csv.vec){
  data.csv <- sub("_unbalanced", "", unb.csv)
  MNIST_dt <- fread(file=data.csv)
  subset_dt <- fread(unb.csv) 
  
  task_dt <- data.table(subset_dt, MNIST_dt)[, odd := factor(y %% 2)]
  feature.names <- grep("^[0-9]+$", names(task_dt), value=TRUE)
  subset.name.vec <- c("seed1_prop0.01","seed2_prop0.1")
  (data.name <- gsub(".*/|[.]csv$", "", unb.csv))
  
  for(subset.name in subset.name.vec){
    subset_vec <- task_dt[[subset.name]]
    task_id <- paste0(data.name,"_",subset.name)
    itask <- mlr3::TaskClassif$new(
      task_id, task_dt[subset_vec != ""], target="odd")
    itask$col_roles$stratum <- "y"
    itask$col_roles$feature <- feature.names
    task.list[[task_id]] <- itask
  }
}


#####
# Create neural network learners
#####
measure_list <- mlr3::msrs(c("classif.prauc", "classif.acc"))
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
    resampling = mlr3::rsmp("cv", folds = 3),
    measure = mlr3::msr("internal_valid_score", minimize = TRUE),
    term_evals = 10,  
    id = id,
    store_models = TRUE,
    store_tuning_instance = TRUE  
  )
}

learning_rates <- c(0.001, 0.005, 0.01)

# Create learners - only AUM now
learner_list <- list(
  lapply(learning_rates, function(lr) {
    make_torch_learner(paste0("linear_AUM_lr", lr), nn_AUM_loss, lr = lr)
  })
) |> unlist(recursive = FALSE)

# Set predict type to prob for all learners
for (i in seq_along(learner_list)) {
  learner_list[[i]]$predict_type <- "prob"
}


#####
# Setup resampling and benchmark grid
#####
kfoldcv <- mlr3::rsmp("cv", folds = 3)

bench.grid <- mlr3::benchmark_grid(
  task.list,
  learner_list,
  kfoldcv
)


#####
# Run benchmark
#####
reg.dir <- "PR-AUM-Imbalanced"
cache.RData <- paste0(reg.dir,".RData")

if(file.exists(cache.RData)){
  load(cache.RData)
}else{
  if(FALSE){#code below only works on the cluster.
    unlink(reg.dir, recursive=TRUE)
    reg = batchtools::makeExperimentRegistry(
      file.dir = reg.dir,
      seed = 1,
      packages = c("mlr3learners","mlr3torch","glmnet")
    )
    mlr3batchmark::batchmark(
      bench.grid, store_models = TRUE, reg=reg)
    job.table <- batchtools::getJobTable(reg=reg)
    chunks <- data.frame(job.table, chunk=1)
    batchtools::submitJobs(chunks, resources=list(
      walltime = 60*60*24,#seconds
      memory = 8000,#megabytes per cpu
      ncpus=1,  #>1 for multicore/parallel jobs.
      ntasks=1, #>1 for MPI jobs
      chunks.as.arrayjobs=TRUE), reg=reg)
    reg <- batchtools::loadRegistry(reg.dir)
    batchtools::getStatus(reg=reg)
    jobs.after <- batchtools::getJobTable(reg=reg)
    extra.cols <- c(algo.pars="learner_id", prob.pars="task_id")
    for(list_col_name in names(extra.cols)){
      new_col_name <- extra.cols[[list_col_name]]
      value <- sapply(jobs.after[[list_col_name]], "[[", new_col_name)
      set(jobs.after, j=new_col_name, value=value)
    }
    table(jobs.after$error)
    exp.ids <- batchtools::findExpired(reg=reg)
    ids <- jobs.after[is.na(error), job.id]
    ok.ids <- setdiff(ids,exp.ids$job.id)
    keep_history <- function(x){
      learners <- x$learner_state$model$marshaled$tuning_instance$archive$learners
      x$learner_state$model <- if(is.function(learners)){
        L <- learners(1)[[1]]
        x$history <- L$model$torch_model_classif$model$callbacks$history
      }
      x
    }
    bench.result <- mlr3batchmark::reduceResultsBatchmark(ok.ids, reg = reg)
  }else{
    ## In the code below, we declare a multisession future plan to
    ## compute each benchmark iteration in parallel on this computer
    ## (data set, learning algorithm, cross-validation fold). For a
    ## few dozen iterations, using the multisession backend is
    ## probably sufficient (I have 12 CPUs on my work PC).
    if(require(future))plan("multisession")
    bench.result <- mlr3::benchmark(bench.grid, store_models = TRUE)
  }
  save(bench.result, file = cache.RData)
}


#####
# Results analysis
#####
score.csv <- paste0(reg.dir,".csv")
score_dt <- bench.result$score(mlr3::msr("classif.prauc"))
score_out <- score_dt[, .(
  task_id, iteration, learner_id, classif.prauc)]
fwrite(score_out, score.csv)

torch_learner_ids <- c("linear_AUM_lr0.001", "linear_AUM_lr0.005", "linear_AUM_lr0.01")
score_torch <- score_dt[
  learner_id %in% torch_learner_ids
][
  , best_epoch := sapply(learner, function(L) {
    if("tuning_result" %in% names(L) && !is.null(L$tuning_result$internal_tuned_values)){
      unlist(L$tuning_result$internal_tuned_values)
    } else {
      NA
    }
  })
][]
names(score_torch)
head(score_torch)

if(nrow(score_torch) > 0){
  history_torch <- score_torch[, {
    L <- learner[[1L]]
    if("archive" %in% names(L) && !is.null(L$archive)){
      M <- L$archive$learners(1)[[1]]$model
      if("torch_model_classif" %in% names(M) && 
         "callbacks" %in% names(M$torch_model_classif$model) &&
         "history" %in% names(M$torch_model_classif$model$callbacks)){
        M$torch_model_classif$model$callbacks$history
      } else {
        data.table()
      }
    } else {
      data.table()
    }
  }, by=.(learner_id, iteration)]
  
  history.csv <- paste0(reg.dir,"_history.csv")
  fwrite(history_torch, history.csv)
}

names(history_torch)
head(history_torch)


#####
# Create visualizations
#####
# Summary statistics
score_stats <- score_dt[, .(
  auc_mean = mean(classif.prauc),
  auc_sd = sd(classif.prauc)
), by=learner_id]

gg <- ggplot()+
  theme_bw()+
  geom_text(aes(
    auc_mean, learner_id, label=sprintf(
      "%.3f±%.3f", auc_mean, auc_sd),
    hjust=fcase(
      auc_mean<0.8, 0,
      auc_mean>0.95, 1,
      default=0.5)),
    vjust=-0.5,
    data=score_stats)+
  geom_point(aes(
    auc_mean, learner_id),
    shape=1,
    data=score_stats)+
  geom_segment(aes(
    auc_mean - auc_sd, learner_id,
    xend=auc_mean + auc_sd, yend=learner_id),
    data=score_stats)+
  scale_x_continuous(
    "Test AUC (mean±SD, 3-fold CV)")

png(paste0(reg.dir, "-test-auc.png"), width=5, height=3.5, units="in", res=200)
print(gg)
dev.off()



if(nrow(score_torch) > 0){
  history_torch <- score_torch[, {
    L <- learner[[1L]]  
    if("tuning_result" %in% names(L) && !is.null(L$tuning_result)){
      # Essayer d'accéder à l'archive du tuning_result
      if("archive" %in% names(L$tuning_result) && !is.null(L$tuning_result$archive)){
        tryCatch({
          M <- L$tuning_result$archive$learners(1)[[1]]$model
          if("torch_model_classif" %in% names(M) && 
             "callbacks" %in% names(M$torch_model_classif$model) &&
             "history" %in% names(M$torch_model_classif$model$callbacks)){
            hist_data <- M$torch_model_classif$model$callbacks$history
            hist_data[, `:=`(task_id = task_id, learner_id = learner_id, iteration = iteration)]
            hist_data
          } else {
            data.table()
          }
        }, error = function(e) {
          data.table()
        })
      } else if("archive" %in% names(L) && !is.null(L$archive)){
        # Alternative : accès direct à l'archive
        tryCatch({
          M <- L$archive$learners(1)[[1]]$model
          if("torch_model_classif" %in% names(M) && 
             "callbacks" %in% names(M$torch_model_classif$model) &&
             "history" %in% names(M$torch_model_classif$model$callbacks)){
            hist_data <- M$torch_model_classif$model$callbacks$history
            hist_data[, `:=`(task_id = task_id, learner_id = learner_id, iteration = iteration)]
            hist_data
          } else {
            data.table()
          }
        }, error = function(e) {
          data.table()
        })
      } else {
        data.table()
      }
    } else {
      data.table()
    }
  }, by=.(task_id, learner_id, iteration)]  # ← IMPORTANT : grouper par task_id aussi !
  
  history.csv <- paste0(reg.dir,"_history_complete.csv")
  fwrite(history_torch, history.csv)
}

names(history_torch)
head(history_torch)

if(exists("history_torch") && nrow(history_torch) > 0){
  history_clean <- history_torch[, .(
    task_id, learner_id, iteration, epoch,
    train.classif.prauc, train.classif.acc,
    valid.classif.prauc, valid.classif.acc
  )]
  history_long <- nc::capture_melt_single(
    history_clean,  # Utiliser la version nettoyée
    set=nc::alevels(valid="validation", train="subtrain"),
    ".classif.",
    measure=nc::alevels(acc="accuracy_prop", prauc="PRAUC"))
  
  if(nrow(history_long) > 0){
    gg_history <- ggplot()+
      theme_bw()+
      geom_line(aes(
        epoch, value, color=set),
        data=history_long[measure=="PRAUC"])+
      facet_grid(rows = vars(task_id), 
                 cols = vars(learner_id, iteration),
                 scales="free_y")+
      scale_y_continuous("PRAUC")+
      scale_x_continuous("epoch")+
      theme(
        strip.text = element_text(size=7),
        strip.text.y = element_text(size=8, angle=0),
        axis.text.x = element_text(angle=45, hjust=1),
        panel.spacing = unit(0.5, "lines")
      )
    
    png(paste0(reg.dir, "-train-auc-by-task-v4.png"), 
        width=15, height=12, units="in", res=200)
    print(gg_history)
    dev.off()
  }
}


#####
# Session Info
#####

sessionInfo()

