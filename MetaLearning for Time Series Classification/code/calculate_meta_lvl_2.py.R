# Title     : Meta feature calculations for time series
# Objective : Generate features for
# Created by: matilda
# Created on: 28.05.20
library("mfe")

path <- "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/calculated_meta_features_split_train_test/train/"
path_store <- "/home/matilda/PycharmProjects/Meta_learning_Time_series_Classification/meta_time_series/meta_data/train/"
all_files <- list.files(path, all.files=TRUE, recursive=TRUE)

calc_mf = path_store
all_calc_mfs <- list.files(calc_mf, all.files=TRUE, recursive=TRUE)

mf <- c("infotheo", "model.based", "statistical", "clustering", "concept", "complexity", "itemset")
mf <- c("statistic")
for (i in seq(1, length(all_files)))
  {
    print(paste("Processing dataset ", all_files[i]))
    data <- read.csv(paste(path, all_files[i], sep=""))
    dimensions <- dim(data)
    dname <- unlist(strsplit(all_files[i], "/"))[-1]
    dname <- paste(dname, "_meta_dataset.csv", sep="")
    name_ <-  paste(path_store, dname, sep="")
    flag = FALSE
    for (j in seq(1, length(all_calc_mfs))){
      if (name_ == paste(calc_mf, all_calc_mfs[j], sep=""))
        {flag = TRUE}

     }
    if (flag == FALSE){
      #y = data[dimensions[2]] + 1
      mf_dataset1 <- metafeatures(data[seq(2, dimensions[2]-1)], data[,dimensions[2]], groups=mf)
      write.csv(mf_dataset1, name_)
      }
  }

