# Create an imbalanced dataset
# Using the code that can be found here : 
# https://tdhock.github.io/blog/2025/imbalance-openml/


library(data.table)
library(mlr3)
library(mlr3pipelines)
library(mlr3torch)
library(mlr3tuning)
library(mlr3learners)
library(sys)



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
  odd = "odd"
)
mtask$col_roles$stratum <- "odd"
mtask$col_roles$feature <- grep("^[0-9]+$", names(MNIST_dt), value = TRUE)



(odd_counts <- MNIST_dt[, .(
  count=.N,
  prop=.N/nrow(MNIST_dt)
), by=odd][order(-prop)])


my.seed <- 1
set.seed(my.seed)
rand_ord <- MNIST_dt[, sample(.N)]
prop_ord <- data.table(y=MNIST_dt$odd[rand_ord])[
  , prop_y := seq(0,1,l=.N), by=y
][, order(prop_y)]
ord_list <- list(
  random=rand_ord,
  proportional=rand_ord[prop_ord])
ord_prop_dt_list <- list()
for(ord_name in names(ord_list)){
  ord_vec <- ord_list[[ord_name]]
  y_ord <- MNIST_dt$odd[ord_vec]
  for(prop_data in c(0.01, 0.1, 1)){
    N <- nrow(MNIST_dt)*prop_data
    N_props <- data.table(y=y_ord[1:N])[, .(
      count=.N,
      prop_y=.N/N
    ), by=y][order(-prop_y)]
    ord_prop_dt_list[[paste(ord_name, prop_data)]] <- data.table(
      ord_name, prop_data, N_props)
  }
}
ord_prop_dt <- rbindlist(ord_prop_dt_list)
dcast(ord_prop_dt, ord_name + prop_data ~ y, value.var="prop_y")


(Tpos <- odd_counts$count[1])
(Tneg <- odd_counts$count[2])

Pneg <- 0.001


(N <- as.integer(Tpos/(3-2*Pneg)))
(n_pos <- Tpos-N)
(n_neg <- as.integer(Pneg*n_pos/(1-Pneg)))


rbind(n_pos+N, Tpos) # all positive labels are used.
rbind(n_neg+N, Tneg) # negative labels used is less than total negative labels.
rbind(2*N, n_pos+n_neg) # imbalanced and balanced subsets are same size.



odd_prop <- 10^seq(-1, -5)
(smaller_dt <- data.table(
  odd_prop,
  N_pos_neg = as.integer(Tpos/(3-2*odd_prop))
)[
  , n_pos := Tpos-N_pos_neg
][
  , n_neg := as.integer(odd_prop*n_pos/(1-odd_prop))
][
  , check_N_im := n_pos+n_neg
][, let(
  check_N_bal = N_pos_neg*2,
  check_prop = n_neg/check_N_im
)][])



subset_mat <- matrix(
  NA, nrow(MNIST_dt), nrow(smaller_dt),
  dimnames=list(
    NULL,
    odd_prop=paste0(
      "seed",
      my.seed,
      "_prop",
      smaller_dt$odd_prop)))
emp_y_list <- list()
emp_props_list <- list()
higgs_ord <- MNIST_dt[, .(odd, .I)][ord_list$proportional]
for(im_i in 1:nrow(smaller_dt)){
  im_row <- smaller_dt[im_i]
  im_count_dt <- im_row[, rbind(
    data.table(subset="b", odd=c(0,1), count=N_pos_neg),
    data.table(subset="i", odd=c(0,1), count=c(n_neg, n_pos))
  )]
  higgs_ord[, subset := NA_character_]
  for(odd_value in c(1,0)){
    tval_dt <- im_count_dt[odd==odd_value]
    sub_vals <- tval_dt[, rep(subset, count)]
    tval_idx <- which(higgs_ord$odd==odd_value)
    some_idx <- tval_idx[1:length(sub_vals)]
    higgs_ord[some_idx, subset := sub_vals]
  }
  subset_mat[higgs_ord$I, im_i] <- higgs_ord$subset
  (im_higgs <- data.table(
    odd_prop=im_row$odd_prop,
    subset=subset_mat[,im_i],
    odd=MNIST_dt$odd
  )[, idx := .I][!is.na(subset)])
  emp_y_list[[im_i]] <- im_higgs[, .(
    count=.N
  ), by=.(odd_prop,subset,odd)]
  emp_props_list[[im_i]] <- im_higgs[, .(
    count=.N,
    first=idx[1], 
    last=idx[.N]
  ), by=.(odd_prop,subset,odd)
  ][
    , prop_in_subset := count/sum(count)
    , by=subset
  ][]
}
emp_y <- rbindlist(emp_y_list)
(emp_props <- rbindlist(emp_props_list))



fwrite(subset_mat, "MNIST_imbalaced.csv")
sys::exec_wait("head","MNIST_imbalaced.csv")
sys::exec_wait("wc", c("-l","MNIST_imbalaced.csv"))
sys::exec_wait("du",c("-k","MNIST_imbalaced.csv"))
