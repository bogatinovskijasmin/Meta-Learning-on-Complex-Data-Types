library(scmamp)
library(jsonlite)
#read the coverage matrix
data<-read.table("../coverage_48x48_6.txt",sep="\t")

#set the threshold to choose the representative data sets for each cluster
threshold<-90
l<-list()
for (i in 1:nrow(data)){
	index<-which(data[i,]>=threshold)
	l[[i]]<-index
}


#read the algorithms results for each dfataset
#set the path where the results are stores
files<-list.files("../agregatedResults",recursive=TRUE,full.names=TRUE)

#extract the alg names
alg_names<-c()
for(i in 1:length(files)){
	alg_names[i]<-strsplit(strsplit(files[i],"agregatedResults/")[[1]][2],"_TESTFOLDS.json")[[1]][1]
}

#read the alg results in a list, one element cosits the results per algorithm
alg_results<-list()
for(i in 1:length(files)){
	alg_results[[i]]<-as.data.frame(fromJSON(files[i]))
}

#data sets that need to be removed from the analysis
ds_remove<-c("Fungi","NonInvasiveFetalECGThorax1","HandOutlines","NonInvasiveFetalECGThorax2","FordB")

#sampling approach
sample_from_each_cluster<-function(sample_size,data,ds_remove){
selected_ds<-c()
ds_names<-colnames(data)
for(i in 1:length(l)){
	if(length(l[[i]])>=sample_size){
		ind<-sample(l[[i]],sample_size)
		while(length(intersect(ds_remove,ds_names[ind]))!=0){
			ind<-sample(l[[i]],sample_size)
		}
		selected_ds<-c(selected_ds,ds_names[ind])
	}
}
return(selected_ds)
}



#check for robustness, the bootstrapping approach
library(PMCMR)

#Sample from each cluster a fixed number of data sets
#set the number of bootstraps
p<-30
#set the sample size
sample_size<-2
mat<-list()
for(k in 1:p){
	selected_ds<-sample_from_each_cluster(sample_size,data,ds_remove)
	M<-matrix(0,length(selected_ds),length(alg_names))
	for(i in 1:length(alg_names)){
		for(j in 1:length(selected_ds)){
			temp<-alg_results[[i]][,which(colnames(alg_results[[i]])==selected_ds[j])]
			if(length(temp)!=0){
			M[j,i]<-alg_results[[i]][,which(colnames(alg_results[[i]])==selected_ds[j])]
			}
			if(length(temp)==0){
			M[j,i]<-100
			}
		
		}
	}
colnames(M)<-alg_names
rownames(M)<-selected_ds

#The next two commands are to save the genrate portfolios
#path_temp<-paste("../Matrix sampling/48_48/100/4/",k,".txt",sep="",collapse="")
#write.table(M,path_temp,sep="\t")

mat[[k]]<-posthoc.friedman.nemenyi.test(M)$p.value
}

alpha<-0.05
for(i in 1:length(mat)){
	temp<-mat[[i]]
	temp[temp<alpha]<-0
 	temp[temp>=alpha]<-1
 	mat[[i]]<-temp
	
}

T<-matrix(0,nrow(mat[[1]]),ncol(mat[[1]]))
for(i in 1:length(mat)){
T<-T+mat[[i]]
	
}


#include all datasets (Table 2 from the paper)
data_names<-colnames(data)

selected_ds<-sample_from_each_cluster(sample_size,data,ds_remove)
M<-matrix(0,length(selected_ds),length(alg_names))
for(i in 1:length(alg_names)){
	for(j in 1:length(selected_ds)){
		temp<-alg_results[[i]][,which(colnames(alg_results[[i]])==selected_ds[j])]
		if(length(temp)!=0){
		M[j,i]<-alg_results[[i]][,which(colnames(alg_results[[i]])==selected_ds[j])]
		}
		if(length(temp)==0){
		M[j,i]<-100
		}
		
	}
}
colnames(M)<-alg_names
rownames(M)<-selected_ds
plotCD(M)

