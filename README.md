

## Introduction
This project is the work I did during my two months summer research internship at the University of Sherbrooke - Department of Computer-Science.

---

## Lexicon
ROC : Receiver Operating Characteristic  
AUC : Area Under the Curve  
AUM : Area Under the Minimum  

PR : Precision-Recall


## Week 1

- I began by redoing the [following code](https://tdhock.github.io/blog/2025/mlr3torch-conv/), only changing the number of epochs from 200 to 2 to make it run on my own computer.  
Results with two epochs :   
<p align="center">
  <img src="Figures/Fig-Week1-2epochs.png" alt="Description" width="800"/>
</p>

&nbsp; Then using an online Jupyterhub cluster I redid the experiment with the intended 200 epochs.      
&nbsp; Here is the result with two hundred epochs :   
<p align="center">
  <img src="Figures/Fig-Week1-200epochs.png" alt="Description" width="800"/>
</p>

- To start learning about the PR curve and its differences with the ROC I redid [this](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/) code and [this](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/) code to better understand the Precision Recall and how to code them in pyhton.

- I also learned about the precision-recall curve by redoing [this code](https://www.blog.trainindata.com/precision-recall-curves/) using the iris dataset (with the Iris-versicolor and the Iris-virginica). I first coded it in [python](https://github.com/juruch/Internship-Sherbrooke/blob/main/Learning-PR-Curves/PrecisionRecall-Iris.ipynb) to follow the tutorial. I then tried to do the same code in R
<p align="center">
  <img src="Figures/Fig-Week1-IrisPR-py.png" alt="Description" width="800"/>
</p>

&nbsp; Here the best treshold is computed using the F1-score :  $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

&nbsp; Where Precision = $\frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}$ and Recall = $\frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}$ = True Positive Rate.

- Finally I did the [comparison between ROC and PR](https://github.com/juruch/Internship-Sherbrooke/blob/main/Learning-PR-Curves/ROCvsPR-BreastCancer.ipynb) curves in python using a dataset a little bigger on breast cancer to see if the size of the dataset would impact the result.
<p align="center">
  <img src="Figures/Fig-Week1-CancerPR-py.png" alt="Description" width="800"/>
</p>
  


## Week 2
- To continue my studies on the PR curve I used [this code](https://tdhock.github.io/blog/2024/auto-grad-overhead/) and transformed it from ROC to PR.
  [Here](ROC to PR Curves/PR.md) I looked at the differences between the two codes.
