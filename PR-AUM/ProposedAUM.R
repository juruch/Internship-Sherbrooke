# Modified AUM to use in archi-PR

Proposed_AUM <- function(pred_tensor, label_tensor) {
  
  is_positive = label_tensor$flatten() == 1
  is_negative = !is_positive
  fn_diff = torch::torch_where(is_positive, -1, 0)
  fp_diff = torch::torch_where(is_positive, 0, 1)
  thresh_tensor = -pred_tensor$flatten()
  sorted_indices = torch::torch_argsort(thresh_tensor)
  fp_denom = torch::torch_sum(is_negative) #or 1 for AUM based on count instead of rate
  fn_denom = torch::torch_sum(is_positive) #or 1 for AUM based on count instead of rate
  sorted_fp_cum = fp_diff[sorted_indices]$cumsum(dim=1)/fp_denom
  sorted_fn_cum = -fn_diff[sorted_indices]$flip(dims=1)$cumsum(dim=1)$flip(dims=1)/fn_denom
  sorted_thresh = thresh_tensor[sorted_indices]
  sorted_is_diff = sorted_thresh$diff() != 0
  sorted_fp_end = torch::torch_cat(list(sorted_is_diff, torch::torch_tensor(TRUE)))
  sorted_fn_end = torch::torch_cat(list(torch::torch_tensor(TRUE), sorted_is_diff))
  uniq_thresh = sorted_thresh[sorted_fp_end]
  uniq_fp_after = sorted_fp_cum[sorted_fp_end]
  uniq_fn_before = sorted_fn_cum[sorted_fn_end]
  FPR = torch::torch_cat(list(torch::torch_tensor(0.0), uniq_fp_after))
  FNR = torch::torch_cat(list(uniq_fn_before, torch::torch_tensor(0.0)))
  
  total_positives = torch::torch_sum(is_positive)
  total_negatives = torch::torch_sum(is_negative)
  TP = total_positives * (1 - FNR)
  TN = total_negatives * (1 - FPR)
  FP = total_negatives * FPR
  FN = total_positives * FNR
  
  precision = torch::torch_where(
    TP + FP == 0,
    torch::torch_tensor(0), 
    TP / (TP + FP)
  )
  #recall = 1 - FNR
  # changed the recall formula to account for the case where (TP+FN) =0
  recall = torch::torch_where(
    TP + FN == 0,
    torch::torch_tensor(0), 
    TP / (TP + FN)
  )
  
  
  pr = list(
    FPR=FPR,
    FNR=FNR,
    accuracy = (TP+TN)/(TP+FP+TN+FN),
    recall=recall,
    precision=precision,
    "min(prec,rec)"=torch::torch_minimum(1 - precision, 1 - recall),
    min_constant=torch::torch_cat(list(torch::torch_tensor(-Inf), uniq_thresh)),
    max_constant=torch::torch_cat(list(uniq_thresh, torch::torch_tensor(Inf))))
  
  
  min_prec_rec = pr[["min(prec,rec)"]][2:-2]
  constant_diff = pr$min_constant[2:N]$diff()
  torch::torch_sum(min_prec_rec * constant_diff)
}