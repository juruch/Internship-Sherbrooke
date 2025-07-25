library(data.table)
library(mlr3)
library(mlr3pipelines)
library(mlr3torch)
library(mlr3tuning)
library(mlr3learners)



#####
# Load AUM loss function
#####


source("PR-Curve.R")

#check if source is working
data.frame(lapply(list.of.tensors, torch::as_array))


nn_PR_AUM <- torch::nn_module(
  "PR_AUM",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward =Proposed_AUM
)



#####
# Read and setup MNIST task
#####


if(!file.exists("MNIST.csv")){
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
# Create neural network learners
#####


measure_list <- msrs(c("classif.prauc", "classif.ce"))
n.epochs <- 200

make_torch_learner <- function(id,loss){
  po_list <- c(
    list(
      mlr3pipelines::po(
        "select",
        selector = mlr3pipelines::selector_type(c("numeric", "integer"))),
      mlr3torch::PipeOpTorchIngressNumeric$new()),
    list(
      mlr3torch::nn("linear",out_features=1),
      mlr3pipelines::po("nn_head"),
      mlr3pipelines::po(
        "torch_loss",
        loss),
      mlr3pipelines::po(
        "torch_optimizer",
        mlr3torch::t_opt("sgd", lr=0.2)),
      mlr3pipelines::po(
        "torch_callbacks",
        mlr3torch::t_clbk("history")),
      mlr3pipelines::po(
        "torch_model_classif",
        batch_size = 100000,
        patience=n.epochs,
        measures_valid=measure_list,
        measures_train=measure_list,
        predict_type="prob",
        epochs = paradox::to_tune(upper = n.epochs, internal = TRUE)))
  )
  graph <- Reduce(mlr3pipelines::concat_graphs, po_list)
  glearner <- mlr3::as_learner(graph)
  mlr3::set_validate(glearner, validate = 0.5)
  mlr3tuning::auto_tuner(
    learner = glearner,
    tuner = mlr3tuning::tnr("internal"),
    resampling = mlr3::rsmp("insample"),
    measure = mlr3::msr("classif.prauc", minimize = TRUE),
    term_evals = 1,
    id=id,
    store_models = TRUE)
}
learner.list<-list(
  make_torch_learner("linear_Cross_entropy",torch::nn_bce_with_logits_loss),
  make_torch_learner("PR_AUM", nn_PR_AUM))



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


reg.dir <- "AUM-test-conv"
cache.RData <- paste0(reg.dir, ".RData")


if(file.exists(cache.RData)){
  load(cache.RData)
}else{
  if(FALSE){#code below only works on the cluster.
    unlink(reg.dir, recursive=TRUE)
    reg = batchtools::makeExperimentRegistry(
      file.dir = reg.dir,
      seed = 1,
      packages = "mlr3verse"
    )
    mlr3batchmark::batchmark(
      bench.grid, store_models = TRUE, reg=reg)
    job.table <- batchtools::getJobTable(reg=reg)
    chunks <- data.frame(job.table, chunk=1)
    batchtools::submitJobs(chunks, resources=list(
      walltime = 2*60*60,#seconds
      memory = 6000,#megabytes per cpu
      ncpus=1,  #>1 for multicore/parallel jobs.
      ntasks=1, #>1 for MPI jobs.
      chunks.as.arrayjobs=TRUE), reg=reg)
    batchtools::getStatus(reg=reg)
    jobs.after <- batchtools::getJobTable(reg=reg)
    table(jobs.after$error)
    ids <- jobs.after[is.na(error), job.id]
    bench.result <- mlr3batchmark::reduceResultsBatchmark(ids, reg = reg)
  }else{
    ## In the code below, we declare a multisession future plan to
    ## compute each benchmark iteration in parallel on this computer
    ## (data set, learning algorithm, cross-validation fold). For a
    ## few dozen iterations, using the multisession backend is
    ## probably sufficient (I have 12 CPUs on my work PC).
    if(require(future))plan("multisession")
    bench.result <- mlr3::benchmark(bench.grid, store_models = TRUE)
  }
  save(bench.result, file=cache.RData)
}



#####
# Error result
#####


score_dt <- bench.result$score()
(score_some <- score_dt[order(classif.ce), .(
  learner_id = factor(learner_id, unique(learner_id)),
  iteration,
  percent_error = 100 * classif.ce)])

library(ggplot2)
ggplot()+
  geom_point(aes(
    percent_error, learner_id),
    shape=1,
    data=score_some)+
  scale_x_continuous(
    breaks=seq(0,70,by=10),
    limits=c(0,70))


(score_stats <- data.table::dcast(
  score_some,
  learner_id ~ .,
  list(mean, sd),
  value.var="percent_error"))


ggplot()+
  geom_point(aes(
    percent_error_mean, learner_id),
    shape=1,
    data=score_stats)+
  geom_segment(aes(
    percent_error_mean+percent_error_sd, learner_id,
    xend=percent_error_mean-percent_error_sd, yend=learner_id),
    data=score_stats)+
  geom_text(aes(
    percent_error_mean, learner_id,
    label=sprintf("%.2f±%.2f", percent_error_mean, percent_error_sd)),
    vjust=-0.5,
    data=score_stats)+
  coord_cartesian(xlim=c(0,70))+  
  scale_x_continuous(
    "Percent error on test set (mean ± SD over 3 folds in CV)",
    breaks=seq(0,70,by=10))


(levs <- levels(score_some$learner_id))


(pval_dt <- data.table(comparison_i=1)[, {
  two_levs <- levs
  lev2rank <- structure(c("lo","hi"), names=two_levs)
  i_long <- score_some[
    learner_id %in% two_levs
  ][
    , rank := lev2rank[paste(learner_id)]
  ][]
      
  i_wide <- dcast(i_long, iteration ~ rank, value.var="percent_error")
  paired <- with(i_wide, t.test(lo, hi, alternative = "l", paired=TRUE))
  unpaired <- with(i_wide, t.test(lo, hi, alternative = "l", paired=FALSE))
  data.table(
      learner.lo=factor(two_levs[1], levs),
      learner.hi=factor(two_levs[2], levs),
      p.paired=paired$p.value,
      p.unpaired=unpaired$p.value,
      mean.diff=paired$estimate,
      mean.lo=unpaired$estimate[1],
      mean.hi=unpaired$estimate[2])
}, by=comparison_i])


ggplot()+
  geom_segment(aes(
    mean.lo, learner.lo,
    xend=mean.hi, yend=learner.lo),
    data=pval_dt,
    size=2,
    color="red")+
  geom_text(aes(
    x=(mean.lo+mean.hi)/2, learner.lo,
    label=sprintf("P=%.4f", p.paired)),
    data=pval_dt,
    vjust=1.5,
    color="red")+
  geom_point(aes(
    percent_error_mean, learner_id),
    shape=1,
    data=score_stats)+
  geom_segment(aes(
    percent_error_mean+percent_error_sd, learner_id,
    xend=percent_error_mean-percent_error_sd, yend=learner_id),
    size=1,
    data=score_stats)+
  geom_text(aes(
    percent_error_mean, learner_id,
    label=sprintf("%.2f±%.2f", percent_error_mean, percent_error_sd)),
    vjust=-0.5,
    data=score_stats)+
  coord_cartesian(xlim=c(0,70))+
  scale_y_discrete("algorithm")+
  scale_x_continuous(
    "Percent error on test set (mean ± SD over 3 folds in CV)",
    breaks=seq(0,70,by=10))



#####
# Validation loss curves
#####


torch_learner_ids <- c("linear_Cross_entropy", "PR_AUM")

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
}, by=.(learner_id, iteration)])


(history_long <- nc::capture_melt_single(
  history_torch,
  set=nc::alevels(valid="validation", train="subtrain"),
  ".classif.",
  measure=nc::alevels("prauc", ce="prop_error")))


ggplot()+
  theme_bw()+
  theme(legend.position=c(0.9, 0.15))+
  geom_vline(aes(
    xintercept=best_epoch),
    data=score_torch)+
  geom_text(aes(
    best_epoch, Inf, label=paste0(" best epoch=", best_epoch)),
    vjust=1.5, hjust=0,
    data=score_torch)+
  geom_line(aes(
    epoch, value, color=set),
    data=history_long[measure=="prauc"])+
  facet_grid(iteration ~ learner_id, labeller=label_both)+
  scale_y_log10("pr auc")+
  scale_x_continuous("epoch")


get_fold <- function(DT,it=1)DT[iteration==it]
history_fold <- get_fold(history_long)
score_fold <- get_fold(score_torch)
min_fold <- history_fold[
  , .SD[value==min(value)]
  , by=.(learner_id, measure, set)
][, point := "min"]
ggplot()+
  theme_bw()+
  theme(legend.position=c(0.9, 0.2))+
  geom_vline(aes(
    xintercept=best_epoch),
    data=score_fold)+
  geom_text(aes(
    best_epoch, Inf, label=paste0(" best epoch=", best_epoch)),
    vjust=1.5, hjust=0,
    data=score_fold)+
  geom_line(aes(
    epoch, value, color=set),
    data=history_fold)+
  geom_point(aes(
    epoch, value, color=set, fill=point),
    shape=21,
    data=min_fold)+
  scale_fill_manual(values=c(min="black"))+
  facet_grid(measure ~ learner_id, labeller=label_both, scales="free")+
  scale_x_continuous("epoch")+
  scale_y_log10("")





#####
# Session Info
#####

sessionInfo()

