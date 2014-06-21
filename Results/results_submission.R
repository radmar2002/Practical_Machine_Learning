rm(list=ls())
setwd("D:/WORK_2014/Certification_Data_Science/Practical_Machine_Learning/Course_Project_Writeup/Results/")
list.files()

#answers = rep("A", 20)

answers <- c("B", "A", "B", "A", "A", 
             "E", "D", "B", "A", "A", 
             "B", "C", "B", "A", "E", 
             "E", "A", "B", "B", "B")

table(answers)

pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

pml_write_files(answers)

list.files()