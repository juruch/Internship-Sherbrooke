# Measures used
# "classif.prauc", "classif.ce", "classif.logloss"

# Using an imbalanced dataset

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

# Check if source is working
data.frame(lapply(list.of.tensors, torch::as_array))

nn_PR_AUM <- torch::nn_module(
  "PR_AUM",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward = Proposed_AUM
)

#####
# Read and setup imbalanced task
#####


imbalanced_dt <- fread("MNIST_imbalanced.csv")

# Explorer la structure des données
print("Structure des données:")
str(imbalanced_dt)
print("Premières lignes:")
head(imbalanced_dt, 10)

# Supposons que vos colonnes sont nommées comme dans l'aperçu fourni
# Adapter selon la structure réelle de vos données

# Vos colonnes sont : seed1_prop0.1, seed1_prop0.01, seed1_prop0.001, seed1_prop1e-04, seed1_prop1e-05
# Choisir la colonne cible (vous pouvez changer selon vos besoins)
target_column <- "seed1_prop0.1"  # Ou une autre colonne selon votre objectif

# Créer une variable binaire pour la classification
imbalanced_dt[, target := ifelse(get(target_column) == "b", 0, 
                                 ifelse(get(target_column) == "i", 1, NA))]

# Supprimer les lignes avec des valeurs manquantes dans la cible
imbalanced_dt <- imbalanced_dt[!is.na(target)]

# Convertir la cible en facteur
imbalanced_dt[, target := factor(target, levels = c(0, 1), labels = c("balanced", "imbalanced"))]

# Option 2: Alternative - si vous voulez utiliser toutes les colonnes comme features
# et créer une tâche différente, décommentez le code ci-dessous:

# # Transformer les données en format long
# imbalanced_long <- melt(imbalanced_dt, 
#                        measure.vars = names(imbalanced_dt),
#                        variable.name = "condition",
#                        value.name = "class")
# 
# # Nettoyer et encoder
# imbalanced_long <- imbalanced_long[class %in% c("b", "i")]
# imbalanced_long[, target := factor(ifelse(class == "b", "balanced", "imbalanced"))]
# 
# # Ajouter des features numériques (exemple: index de ligne, condition encodée)
# imbalanced_long[, row_id := .I]
# imbalanced_long[, condition_encoded := as.numeric(as.factor(condition))]

# Vérifier l'équilibre des classes
print("Distribution des classes:")
table(imbalanced_dt$target)
print("Proportions:")
prop.table(table(imbalanced_dt$target))

# Créer des features numériques à partir des autres colonnes
feature_cols <- setdiff(names(imbalanced_dt), c(target_column, "target"))

# Encoder les colonnes catégorielles en numériques
# Avec vos 5 colonnes : seed1_prop0.1, seed1_prop0.01, seed1_prop0.001, seed1_prop1e-04, seed1_prop1e-05
for(col in feature_cols) {
  imbalanced_dt[, paste0(col, "_b") := as.numeric(get(col) == "b")]
  imbalanced_dt[, paste0(col, "_i") := as.numeric(get(col) == "i")]
  imbalanced_dt[, paste0(col, "_missing") := as.numeric(is.na(get(col)) | get(col) == "")]
}

# Identifier les features numériques créées
numeric_features <- grep("_(b|i|missing)$", names(imbalanced_dt), value = TRUE)

# Créer la tâche MLR3
mtask <- mlr3::TaskClassif$new(
  id = "ImbalancedClassif",
  backend = imbalanced_dt,
  target = "target"
)

# Définir les rôles des colonnes
mtask$col_roles$stratum <- "target"
mtask$col_roles$feature <- numeric_features

print("Informations sur la tâche:")
print(mtask)
print("Features utilisées:")
print(mtask$feature_names)

#####
# Create neural network learners
#####

measure_list <- msrs(c("classif.prauc", "classif.ce", "classif.logloss"))
n.epochs <- 500

make_torch_learner <- function(id, loss) {
  po_list <- c(
    list(
      mlr3pipelines::po(
        "select",
        selector = mlr3pipelines::selector_type(c("numeric", "integer"))),
      mlr3torch::PipeOpTorchIngressNumeric$new()),
    list(
      mlr3torch::nn("linear", out_features = 1),
      mlr3pipelines::po("nn_head"),
      mlr3pipelines::po(
        "torch_loss",
        loss),
      mlr3pipelines::po(
        "torch_optimizer",
        mlr3torch::t_opt("sgd", lr = 0.01)),
      mlr3pipelines::po(
        "torch_callbacks",
        mlr3torch::t_clbk("history")),
      mlr3pipelines::po(
        "torch_model_classif",
        batch_size = min(1000, nrow(imbalanced_dt)),  # Adapter à la taille des données
        patience = n.epochs,
        measures_valid = measure_list,
        measures_train = measure_list,
        predict_type = "prob",
        epochs = paradox::to_tune(upper = n.epochs, internal = TRUE)))
  )
  graph <- Reduce(mlr3pipelines::concat_graphs, po_list)
  glearner <- mlr3::as_learner(graph)
  mlr3::set_validate(glearner, validate = 0.5)
  mlr3tuning::auto_tuner(
    learner = glearner,
    tuner = mlr3tuning::tnr("internal"),
    resampling = mlr3::rsmp("insample"),
    measure = mlr3::msr("classif.prauc", minimize = FALSE),  # PRAUC doit être maximisé
    term_evals = 1,
    id = id,
    store_models = TRUE)
}

learner.list <- list(
  make_torch_learner("linear_Cross_entropy", torch::nn_bce_with_logits_loss),
  make_torch_learner("PR_AUM", nn_PR_AUM))

#####
# Benchmark grid
#####

# Benchmark grid - la stratification se fait automatiquement si stratum est défini dans la tâche
kfoldcv <- rsmp("cv", folds = 3)

# Vérifier que la stratification est bien configurée dans la tâche
print("Stratum column dans mtask:")
print(mtask$col_roles$stratum)

bench.grid <- benchmark_grid(
  mtask,
  learner.list,
  kfoldcv
)

#####
# Run benchmark
#####

reg.dir <- "test-imbalanced"
cache.RData <- paste0(reg.dir, ".RData")

if(file.exists(cache.RData)) {
  load(cache.RData)
} else {
  if(FALSE) {  # code below only works on the cluster.
    unlink(reg.dir, recursive = TRUE)
    reg = batchtools::makeExperimentRegistry(
      file.dir = reg.dir,
      seed = 1,
      packages = "mlr3verse"
    )
    mlr3batchmark::batchmark(
      bench.grid, store_models = TRUE, reg = reg)
    job.table <- batchtools::getJobTable(reg = reg)
    chunks <- data.frame(job.table, chunk = 1)
    batchtools::submitJobs(chunks, resources = list(
      walltime = 2*60*60,  # seconds
      memory = 6000,  # megabytes per cpu
      ncpus = 1,  # >1 for multicore/parallel jobs.
      ntasks = 1, # >1 for MPI jobs.
      chunks.as.arrayjobs = TRUE), reg = reg)
    batchtools::getStatus(reg = reg)
    jobs.after <- batchtools::getJobTable(reg = reg)
    table(jobs.after$error)
    ids <- jobs.after[is.na(error), job.id]
    bench.result <- mlr3batchmark::reduceResultsBatchmark(ids, reg = reg)
  } else {
    ## Compute each benchmark iteration in parallel
    if(require(future)) plan("multisession")
    bench.result <- mlr3::benchmark(bench.grid, store_models = TRUE)
  }
  save(bench.result, file = cache.RData)
}

#####
# Error result
#####

score_dt <- bench.result$score()

# Vérifier quelles colonnes sont disponibles
print("Colonnes disponibles dans score_dt:")
print(names(score_dt))

# Identifier les métriques de classification disponibles
metric_cols <- grep("^classif\\.", names(score_dt), value = TRUE)
print("Métriques de classification disponibles:")
print(metric_cols)

# Créer score_some avec les métriques disponibles
if("classif.prauc" %in% names(score_dt)) {
  (score_some <- score_dt[order(classif.ce), .(
    learner_id = factor(learner_id, unique(learner_id)),
    iteration,
    percent_error = 100 * classif.ce,
    prauc = classif.prauc)])
} else {
  print("classif.prauc non disponible, utilisation des métriques disponibles")
  (score_some <- score_dt[order(classif.ce), .(
    learner_id = factor(learner_id, unique(learner_id)),
    iteration,
    percent_error = 100 * classif.ce)])
}

library(ggplot2)

# Graphique des erreurs seulement
ggplot()+
  geom_point(aes(
    percent_error, learner_id),
    shape = 1,
    data = score_some)+
  scale_x_continuous(
    breaks = seq(0, 100, by = 10),
    limits = c(0, 100))

# Note: PRAUC n'est pas disponible dans score_dt, mais visible dans l'historique d'entraînement

# Statistiques résumées (seulement pour percent_error)
(score_stats <- data.table::dcast(
  score_some,
  learner_id ~ .,
  list(mean, sd),
  value.var = "percent_error"))

# Graphique avec barres d'erreur
ggplot()+
  geom_point(aes(
    percent_error_mean, learner_id),
    shape = 1,
    data = score_stats)+
  geom_segment(aes(
    percent_error_mean + percent_error_sd, learner_id,
    xend = percent_error_mean - percent_error_sd, yend = learner_id),
    data = score_stats)+
  geom_text(aes(
    percent_error_mean, learner_id,
    label = sprintf("%.2f±%.2f", percent_error_mean, percent_error_sd)),
    vjust = -0.5,
    data = score_stats)+
  coord_cartesian(xlim = c(0, 100))+  
  scale_x_continuous(
    "Percent error on test set (mean ± SD over 3 folds in CV)",
    breaks = seq(0, 100, by = 10))

# Test statistique
(levs <- levels(score_some$learner_id))

if(length(levs) >= 2) {
  (pval_dt <- data.table(comparison_i = 1)[, {
    two_levs <- levs[1:2]  # Prendre les deux premiers learners
    lev2rank <- structure(c("lo", "hi"), names = two_levs)
    i_long <- score_some[
      learner_id %in% two_levs
    ][
      , rank := lev2rank[paste(learner_id)]
    ][]
    
    i_wide <- dcast(i_long, iteration ~ rank, value.var = "percent_error")
    paired <- with(i_wide, t.test(lo, hi, alternative = "l", paired = TRUE))
    unpaired <- with(i_wide, t.test(lo, hi, alternative = "l", paired = FALSE))
    data.table(
      learner.lo = factor(two_levs[1], levs),
      learner.hi = factor(two_levs[2], levs),
      p.paired = paired$p.value,
      p.unpaired = unpaired$p.value,
      mean.diff = paired$estimate,
      mean.lo = unpaired$estimate[1],
      mean.hi = unpaired$estimate[2])
  }, by = comparison_i])
  
  # Graphique de comparaison
  ggplot()+
    geom_segment(aes(
      mean.lo, learner.lo,
      xend = mean.hi, yend = learner.lo),
      data = pval_dt,
      linewidth = 2,  # Correction pour ggplot2 récent
      color = "red")+
    geom_text(aes(
      x = (mean.lo + mean.hi)/2, learner.lo,
      label = sprintf("P=%.4f", p.paired)),
      data = pval_dt,
      vjust = 1.5,
      color = "red")+
    geom_point(aes(
      percent_error_mean, learner_id),
      shape = 1,
      data = score_stats)+
    geom_segment(aes(
      percent_error_mean + percent_error_sd, learner_id,
      xend = percent_error_mean - percent_error_sd, yend = learner_id),
      linewidth = 1,  # Correction pour ggplot2 récent
      data = score_stats)+
    geom_text(aes(
      percent_error_mean, learner_id,
      label = sprintf("%.2f±%.2f", percent_error_mean, percent_error_sd)),
      vjust = -0.5,
      data = score_stats)+
    coord_cartesian(xlim = c(0, 100))+
    scale_y_discrete("algorithm")+
    scale_x_continuous(
      "Percent error on test set (mean ± SD over 3 folds in CV)",
      breaks = seq(0, 100, by = 10))
}

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
}, by = .(learner_id, iteration)])

# Capturer toutes les métriques disponibles
(history_long <- nc::capture_melt_single(
  history_torch,
  set = nc::alevels(valid = "validation", train = "subtrain"),
  ".classif.",
  measure = nc::alevels("prauc", "ce", "logloss")))

# Graphique des courbes d'apprentissage
ggplot()+
  theme_bw()+
  theme(legend.position = c(0.9, 0.15))+
  geom_vline(aes(
    xintercept = best_epoch),
    data = score_torch)+
  geom_text(aes(
    best_epoch, Inf, label = paste0(" best epoch=", best_epoch)),
    vjust = 1.5, hjust = 0,
    data = score_torch)+
  geom_line(aes(
    epoch, value, color = set),
    data = history_long)+
  facet_grid(measure ~ learner_id + iteration, labeller = label_both, scales = "free")+
  scale_y_log10("Metric value")+
  scale_x_continuous("epoch")+
  theme(legend.position.inside = c(0.9, 0.15))  # Correction pour ggplot2 récent

# Analyser les résultats PRAUC depuis l'historique
# Extraire les valeurs PRAUC finales pour chaque modèle
prauc_final <- history_torch[, .SD[epoch == max(epoch)], by = .(learner_id, iteration)][
  , .(learner_id, iteration, 
      train_prauc = train.classif.prauc, 
      valid_prauc = valid.classif.prauc)]

print("PRAUC final par modèle:")
print(prauc_final)

# Moyennes PRAUC par algorithme
prauc_summary <- prauc_final[, .(
  mean_train_prauc = mean(train_prauc),
  sd_train_prauc = sd(train_prauc),
  mean_valid_prauc = mean(valid_prauc),
  sd_valid_prauc = sd(valid_prauc)
), by = learner_id]

print("Résumé PRAUC:")
print(prauc_summary)

#####
# Session Info
#####

sessionInfo()