args <- commandArgs(TRUE)
input_file<-args[1]
output_file<-args[2]

dim1<-args[3]
dim2<-args[4]

files<-list.files("ts_fresh_lvl1_full/Univariate2018_ts/Univariate_ts",full.names=TRUE,recursive=TRUE)

data<-list()
for (i in 1:length(files)){
    data[[i]]<-read.ts(files[i],header=TRUE,sep=",")
}

names<-c()
for(i in 1:length(files)){
    names[i]<-strsplit(strsplit(files[i],"Univariate_ts/")[[1]][2],"/")[[1]][1]
}

count<-c()
for(i in 1:length(data)){
    count[i]<-nrow(data[[i]])
}

df_counts<-cbind(names,count)

cell_inst<-read.table("cell_instance.txt",sep="\t")


coverage_matrix <- function(file, output_file) {
    clusters<-read.csv(file,sep="\t",header=FALSE)
    clusters<-cbind(clusters,seq(1,dim1*dim2,1))
    colnames(clusters)<-c("groupes","cell")
    
    clust<-c()
    for( i in 1:nrow(cell_inst)){
        clust[i]<-clusters$groupes[which(clusters$cell==cell_inst$t[i])]

    }


    df1<-cbind(cell_inst,clust)
    colnames(df1)<-c("name","cell","clust")
    
    l<-list()
    counter<-1
    for(i in sort(unique(clust))){
        index<-which(df1$clust==i)
        temp<-df1[index,]


        y<-c()
        for(j in 1:nrow(temp)){
            y[j]<-strsplit(temp$name[j],"_")[[1]][1]
        }
        l[[counter]]<-unique(y)
        counter<-counter+1
        
        coverage<-matrix(0,length(unique(clust)),113)
        counter<-1
        for(i in sort(unique(clust))){
            index<-which(df1$clust==i)
            temp<-df1[index,]
            y<-c()
            for(j in 1:nrow(temp)){
                y[j]<-strsplit(temp$name[j],"_")[[1]][1]
            }

            for(k in 1:nrow(df_counts)){
                coverage[counter,k]<-length(which(y==df_counts[k,1]))/as.numeric(as.character(df_counts[k,2]))*100
            }
            counter<-counter+1
        }
    }
    
    #m<-basename(file)
    rownames(coverage)<-paste("Cluster", seq(1,length(unique(clust)),1))
    colnames(coverage)<-names

    #dir<-"coverage_matrix/"
    #path<-paste(dir, m)
    #dir.create(dir)
    write.table(coverage, output_file, sep="\t")
}



coverage_matrix(input_file, output_file)