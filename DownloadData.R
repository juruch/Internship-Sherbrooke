install.packages("googledrive")
library(googledrive)

if (!file.exists("MNIST.csv")) {
  options(timeout = 600)
  file_id <- "1-uFJyJIWZwD5Z58Zzo1XeO8CB45gXO6q"
  drive_download(as_id(file_id), path = "MNIST.csv", overwrite = TRUE)
}
if (!file.exists("EMNIST.csv")) {
  options(timeout = 600)
  file_id <- "1oH5xqYAoFxQ-jUUAnwPK4_TBc5ZkUUt0"
  drive_download(as_id(file_id), path = "EMNIST.csv", overwrite = TRUE)
}
if (!file.exists("FashionMNIST.csv")) {
  options(timeout = 600)
  file_id <- "13XsdIj7vefmBkJaW2RaNT2RYsrV-4P4p"
  drive_download(as_id(file_id), path = "FashionMNIST.csv", overwrite = TRUE)
}
