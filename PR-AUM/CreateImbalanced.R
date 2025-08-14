# Create an imbalanced dataset
# Using the code that can be found here : 
# https://tdhock.github.io/blog/2025/unbalanced/


library(data.table)
data_Classif <- "~/data_Classif/"
for(data.name in c("EMNIST", "FashionMNIST", "MNIST")){
  data.csv <- paste0(
    data_Classif,
    data.name,
    ".csv")
  MNIST_dt <- fread(data.csv)
  seed_mat_list <- list()
  for(seed in 1:2){
    set.seed(seed)
    rand_ord <- MNIST_dt[, sample(.N)]
    prop_ord <- data.table(y=MNIST_dt$y[rand_ord])[
      , prop_y := seq(0,1,l=.N), by=y
    ][, order(prop_y)]
    ord_list <- list(
      random=rand_ord,
      proportional=rand_ord[prop_ord])
    (binary_counts <- MNIST_dt[
      , odd := y %% 2
    ][, .(
      count=.N,
      prop=.N/nrow(MNIST_dt)
    ), by=odd][order(-prop, -odd)])
    larger_N <- binary_counts$count[1]/2
    target_prop <- c(0.5, 0.1, 0.05, 0.01, 0.005, 0.001)
    (smaller_dt <- data.table(
      target_prop,
      count=as.integer(target_prop*larger_N/(1-target_prop))
    )[
      , prop := count/(count+larger_N)
    ][])
    (unb_small_dt <- data.table(
      subset="unbalanced",
      binary_counts[2,.(odd)],
      smaller_dt[-1]))
    subset_mat <- matrix(
      NA, nrow(MNIST_dt), nrow(unb_small_dt),
      dimnames=list(
        NULL,
        target_prop=paste0(
          "seed",
          seed,
          "_prop",
          unb_small_dt$target_prop)))
    emp_y_list <- list()
    emp_props_list <- list()
    MNIST_ord <- MNIST_dt[, .(odd, .I)][ord_list$proportional]
    for(unb_i in 1:nrow(unb_small_dt)){
      unb_row <- unb_small_dt[unb_i]
      unb_count_dt <- rbind(
        data.table(subset="balanced", binary_counts[,.(odd)], smaller_dt[1]),
        data.table(subset="unbalanced", binary_counts[1,.(odd)], smaller_dt[1]),
        unb_row)
      MNIST_ord[, subset := NA_character_]
      for(o in c(1,0)){
        o_dt <- unb_count_dt[odd==o]
        sub_vals <- o_dt[, rep(subset, count)]
        o_idx <- which(MNIST_ord$odd==o)
        some_idx <- o_idx[1:length(sub_vals)]
        MNIST_ord[some_idx, subset := sub_vals]
      }
      subset_mat[MNIST_ord$I, unb_i] <- MNIST_ord$subset
      ## Check to make unbalanced is a subset of the previous larger one.
      if(unb_i>1)stopifnot(all(which(
        subset_mat[,unb_i]=="unbalanced"
      ) %in% which(
        subset_mat[,unb_i-1]=="unbalanced"
      )))
      ## Check to make sure balanced is the same as previous.
      if(unb_i>1)stopifnot(identical(
        which(subset_mat[,unb_i]=="balanced"),
        which(subset_mat[,unb_i-1]=="balanced")
      ))
      (unb_MNIST <- data.table(
        target_prop=unb_row$target_prop,
        subset=subset_mat[,unb_i],
        odd=MNIST_dt$odd,
        y=MNIST_dt$y)[, idx := .I][!is.na(subset)])
      emp_y_list[[unb_i]] <- unb_MNIST[, .(
        count=.N
      ), by=.(target_prop,subset,y)]
      emp_props_list[[unb_i]] <- unb_MNIST[, .(
        count=.N,
        first=idx[1], 
        last=idx[.N]
      ), keyby=.(target_prop,subset,odd)
      ][
        , prop_in_subset := count/sum(count)
        , by=subset
      ][]
    }
    emp_y <- rbindlist(emp_y_list)
    (emp_props <- rbindlist(emp_props_list))
    seed_mat_list[[seed]] <- subset_mat
  }
  print(data.name)
  print(dcast(emp_y, subset + target_prop ~ y, value.var="count"))
  (seed_dt <- do.call(data.table, seed_mat_list))
  (out.csv <- sub("data_Classif", "data_Classif_unbalanced", data.csv))
  dir.create(dirname(out.csv), showWarnings = FALSE, recursive = FALSE)
  fwrite(seed_dt, out.csv)
}
