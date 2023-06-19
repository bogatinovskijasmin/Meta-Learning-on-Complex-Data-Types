install.packages("aweSOM", repos='http://cran.us.r-project.org')

library(aweSOM)

set.seed(1465)

args <- commandArgs(TRUE)
features_file<-args[1]
output_file<-args[2]

dim1<-strtoi(args[3])
dim2<-strtoi(args[4])
iters<-strtoi(args[5])

print("Load data")
data<-read.table(features_file, sep="\t",header=TRUE)
data_temp<-scale(data.matrix(data))


print("Train SOM")
init<-somInit(data_temp, dim1, dim2)
som.model <- kohonen::som(data_temp, grid = kohonen::somgrid(dim1, dim2, "hexagonal"), rlen = iters, alpha = c(0.05, 0.01), radius = c(2.65,-2.65), dist.fcts = "sumofsquares", init = init)

x<-somQuality(som.model, data_temp) 

write.table(som.model$codes, output_file)