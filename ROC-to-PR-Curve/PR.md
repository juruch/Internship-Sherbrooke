# Comparison between ROC and PR

[The original code for ROC](https://github.com/tdhock/tdhock.github.io/blob/master/_posts/2024-10-28-auto-grad-overhead.md) 
with documentation and explenations.

---
Here you can find the original code for the ROC and the translation I did for the PR.
I also entered the result for both codes to compare the outputs.


# Curve

``` r
four_labels_vec <- c(-1,-1,1,1)
four_pred_vec <- c(2.0, -3.5, -1.0, 1.5)
``` 

``` r
ROC_curve <- function(pred_tensor, label_tensor){
  is_positive = label_tensor == 1
  is_negative = label_tensor != 1
  fn_diff = torch::torch_where(is_positive, -1, 0)
  fp_diff = torch::torch_where(is_positive, 0, 1)
  thresh_tensor = -pred_tensor$flatten()
  sorted_indices = torch::torch_argsort(thresh_tensor)
  fp_denom = torch::torch_sum(is_negative) #or 1 for AUM based on count instead of rate
  fn_denom = torch::torch_sum(is_positive) #or 1 for AUM based on count instead of rate
  sorted_fp_cum = fp_diff[sorted_indices]$cumsum(dim=1)/fp_denom
  sorted_fn_cum = -fn_diff[sorted_indices]$flip(1)$cumsum(dim=1)$flip(1)/fn_denom
  sorted_thresh = thresh_tensor[sorted_indices]
  sorted_is_diff = sorted_thresh$diff() != 0
  sorted_fp_end = torch::torch_cat(c(sorted_is_diff, torch::torch_tensor(TRUE)))
  sorted_fn_end = torch::torch_cat(c(torch::torch_tensor(TRUE), sorted_is_diff))
  uniq_thresh = sorted_thresh[sorted_fp_end]
  uniq_fp_after = sorted_fp_cum[sorted_fp_end]
  uniq_fn_before = sorted_fn_cum[sorted_fn_end]
  FPR = torch::torch_cat(c(torch::torch_tensor(0.0), uniq_fp_after))
  FNR = torch::torch_cat(c(uniq_fn_before, torch::torch_tensor(0.0)))
  list(
    FPR=FPR,
    FNR=FNR,
    TPR=1 - FNR,
    "min(FPR,FNR)"=torch::torch_minimum(FPR, FNR),
    min_constant=torch::torch_cat(c(torch::torch_tensor(-Inf), uniq_thresh)),
    max_constant=torch::torch_cat(c(uniq_thresh, torch::torch_tensor(Inf))))
}
four_labels <- torch::torch_tensor(four_labels_vec)
four_pred <- torch::torch_tensor(four_pred_vec)
list.of.tensors <- ROC_curve(four_pred, four_labels)
data.frame(lapply(list.of.tensors, torch::as_array))
```

``` r
PR_curve <- function(pred_tensor, label_tensor){
  is_positive = label_tensor == 1
  is_negative = label_tensor != 1
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
  FP = total_negatives * FPR
  FN = total_positives * FNR
    
  precision = torch::torch_where(
    TP + FP == 0,
    torch::torch_tensor(NaN), 
    TP / (TP + FP)
  )
  #recall = 1 - FNR
  # changed the recall formula to account for the case where (TP+FN) = 0
  recall = torch::torch_where(
    TP + FN == 0,
    torch::torch_tensor(NaN), 
    TP / (TP + FN)
  )
  
  list(
    FPR=FPR,
    FNR=FNR,
    recall=recall,
    precision=precision,
    "min(prec,rec)"=torch::torch_minimum(1 - precision, 1 - recall),
    min_constant=torch::torch_cat(list(torch::torch_tensor(-Inf), uniq_thresh)),
    max_constant=torch::torch_cat(list(uniq_thresh, torch::torch_tensor(Inf))))
}

four_labels <- torch::torch_tensor(four_labels_vec)
four_pred <- torch::torch_tensor(four_pred_vec)
list.of.tensors <- PR_curve(four_pred, four_labels)
data.frame(lapply(list.of.tensors, torch::as_array))
```

In the PR_curve function, the precision and recall are computed in a way that handles the cases where (TP+FP)=0 or (TP+FN)=0. In those cases the precision and/or the recall is set to NaN.

<br />

**ROC result :**   
```
##   FPR FNR TPR min.FPR.FNR. min_constant max_constant
## 1 0.0 1.0 0.0          0.0         -Inf         -2.0
## 2 0.5 1.0 0.0          0.5         -2.0         -1.5
## 3 0.5 0.5 0.5          0.5         -1.5          1.0
## 4 0.5 0.0 1.0          0.0          1.0          3.5
## 5 1.0 0.0 1.0          0.0          3.5          Inf
```
> The table above also has one row for each point on the ROC curve, and the following columns:
* `FPR` = $\frac{\text{False Positive}}{\text{False Positive} + \text{True Negative}}$, the False Positive Rate (X axis of ROC curve plot), 
* `TPR` $\frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}$, the True Positive Rate (Y axis of ROC curve plot), 
* `FNR=1-TPR` is the False Negative Rate,
* `min(FPR,FNR)` is the minimum of `FPR` and `FNR`,
* and `min_constant`, `max_constant` give the range of constants which result in the corresponding error values. For example, the second row means that adding any constant between -2 and -1.5 results in predicted classes that give FPR=0.5 and TPR=0.

**PR result :**  
```
##   FPR FNR   recall precision   min.prec.rec.  min_constant  max_constant
## 1 0.0 1.0    0.0       NaN           NaN         -Inf         -2.0
## 2 0.5 1.0    0.0 0.0000000           1.0         -2.0         -1.5
## 3 0.5 0.5    0.5 0.5000000           0.5         -1.5          1.0
## 4 0.5 0.0    1.0 0.6666667           0.0          1.0          3.5
## 5 1.0 0.0    1.0 0.5000000           0.0          3.5          Inf
```
* `Precision` = $\frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}$
* `Recall` = $\frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}$ = True Positive Rate.
* `min(prec,rec)` the minimum between `1 - Precision` and `1 - Recall`

<br />

In the PR results we see a case where precision=NaN because (TP+FP)=0.


 ---
 
# AUC (Area Under the Curve)

``` r
ROC_AUC <- function(pred_tensor, label_tensor){
  roc = ROC_curve(pred_tensor, label_tensor)
  FPR_diff = roc$FPR[2:N]-roc$FPR[1:-2]
  TPR_sum = roc$TPR[2:N]+roc$TPR[1:-2]
  torch::torch_sum(FPR_diff*TPR_sum/2.0)
}
ROC_AUC(four_pred, four_labels)
```

``` r
PR_AUC <- function(pred_tensor, label_tensor){
  pr = PR_curve(pred_tensor, label_tensor)
  recall_diff = pr$recall[3:N]-pr$recall[2:-2]
  precision_sum = pr$precision[3:N]+pr$precision[2:-2]
  torch::torch_sum(recall_diff*precision_sum/2.0)
}
PR_AUC(four_pred, four_labels)
```
I changed the number of values taken by pr$precision and pr$recall to not take the first line where precision=NaN.

**ROC result :** 
```
## torch_tensor
## 0.5
## [ CPUFloatType{} ]
```

**PR result :** 
```
## torch_tensor
## 0.416667
## [ CPUFloatType{} ]
```

> The ROC AUC value is 0.5, indicating that `four_pred` is not a very
good vector of predictions with respect to `four_labels`.


---
# AUM (Area Under the Minimum)
> Recently [in JMLR23](https://jmlr.org/papers/v24/21-0751.html)
we proposed a new loss function called the AUM, Area Under Min of
False Positive and False Negative rates. We showed that is can be
interpreted as a L1 relaxation of the sum of min of False Positive and
False Negative rates, over all points on the ROC curve. We
additionally showed that AUM is piecewise linear, and differentiable
almost everywhere, so can be used in gradient descent learning
algorithms. Finally, we showed that minimizing AUM encourages points
on the ROC curve to move toward the upper left, thereby encouraging
large AUC. Computation of the AUM loss requires first
computing ROC curves (same as above), as in the code below.


``` r
Proposed_AUM <- function(pred_tensor, label_tensor){
  roc = ROC_curve(pred_tensor, label_tensor)
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2]
  constant_diff = roc$min_constant[2:N]$diff()
  torch::torch_sum(min_FPR_FNR * constant_diff)
}
```

``` r
Proposed_AUM <- function(pred_tensor, label_tensor){
  pr = PR_curve(pred_tensor, label_tensor)
  min_prec_rec = pr[["min(prec,rec)"]][2:-2]
  constant_diff = pr$min_constant[2:N]$diff()
  torch::torch_sum(min_prec_rec * constant_diff)
}
```
<br />

``` r
four_pred$requires_grad <- TRUE
(four_aum <- Proposed_AUM(four_pred, four_labels))
```

**ROC result :** 
```
## torch_tensor
## 1.5
## [ CPUFloatType{} ][ grad_fn = <SumBackward0> ]
```

**PR result :** 
```
## torch_tensor
## 1.75
## [ CPUFloatType{} ][ grad_fn = <SumBackward0> 
```


<br />

``` r
four_pred$grad
```

**ROC result :** 
```
## torch_tensor
## [ Tensor (undefined) ]
```

**PR result :** 
```
## torch_tensor
## [ Tensor (undefined) ]
```

<br />

``` r
four_aum$backward()
four_pred$grad
```

**ROC result :**  
```
## torch_tensor
##  0.5000
## -0.0000
## -0.5000
## -0.0000
## [ CPUFloatType{4} ]
```

**PR result :**  
```
## torch_tensor
##  1.0000
## -0.0000
## -0.5000
## -0.5000
## [ CPUFloatType{4} ]
```
