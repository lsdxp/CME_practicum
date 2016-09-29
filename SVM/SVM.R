setwd("C:/jwang41/CME")
library(e1071)

#a function to round all the data to 2 position behind decimal point
rounddata=function(data)    
{
  for(i in 1:ncol(data))
  {
    data[,i]=round(data[,i],2)
  }
  data
}

#a function to pick the trade price change that is not 0
filter_data=function(data)             
{
  data[data[,ncol(data)-1]!=0,]
}

#a function to run SVM first 80 predict last 20 and last 20 predict first 80
run_svm_80_20=function(data)         
{
  trade_price=data[,ncol(data)]
  datadata=data[,-ncol(data)]
  b=datadata[,ncol(datadata)];
  b=as.factor(b);
  
  
  percent80=round(0.8*nrow(datadata));
  
  trade_price1=trade_price[(percent80+1):length(b)];
  trade_price2=trade_price[1:percent80];
  
  #create testing sets and training sets
  data1x=datadata[1:percent80,2:(ncol(datadata)-1)];   
  data1y=b[1:percent80];
  
  data2x=datadata[(percent80+1):nrow(datadata),2:(ncol(datadata)-1)];
  data2y=b[(percent80+1):length(b)];
  
  real1=data2y;
  real2=data1y;
  
  #build-in function for SVM in R
  model1=svm(data1x,data1y,kernel="radial");         
  model2=svm(data2x,data2y,kernel ="radial");                                     
  
  
  
  predict1=predict(model1,data2x);   
  predict2=predict(model2,data1x);
  
  a=list();
  #return to a list in R, first 2 element is the result table for 20-80 and 80-20
  a[[1]]=table(real1,predict1);    
  a[[2]]=table(real2,predict2);
  
  #3rd and 4th elements are all the testing data and prediction
  a[[3]]=data.frame(data2x,real1,predict1,trade_price1);  
  a[[4]]=data.frame(data1x,real2,predict2,trade_price2);
  a
}

#a function that is same with the 20_80 one, except training/testing is 50/50
run_svm_50_50=function(data)    
{
  trade_price=data[,ncol(data)]
  datadata=data[,-ncol(data)]
  b=datadata[,ncol(datadata)];
  b=as.factor(b);
  
  
  percent50=round(0.5*nrow(datadata));
  
  trade_price1=trade_price[(percent50+1):length(b)];
  trade_price2=trade_price[1:percent50];
  
  data1x=datadata[1:percent50,2:(ncol(datadata)-1)];
  data1y=b[1:percent50];
  
  data2x=datadata[(percent50+1):nrow(datadata),2:(ncol(datadata)-1)];
  data2y=b[(percent50+1):length(b)];
  
  real1=data2y;
  real2=data1y;
  
  model1=svm(data1x,data1y,kernel="radial");
  model2=svm(data2x,data2y,kernel ="radial");                                     
  
  
  
  predict1=predict(model1,data2x);
  predict2=predict(model2,data1x);
  
  a=list();
  a[[1]]=table(real1,predict1);
  a[[2]]=table(real2,predict2);
  #output is the same as 80-20 function
  a[[3]]=data.frame(data2x,real1,predict1,trade_price1);        
  a[[4]]=data.frame(data1x,real2,predict2,trade_price2);
  a
}


data1 <-read.table("with1BookRec.txt",sep=",",header=TRUE)
data3 <-read.table("with3BookRec.txt",sep=",",header=TRUE)

data1=rounddata(data1)
data3=rounddata(data3)

#first round and filter data
data1=filter_data(data1)      
data3=filter_data(data3)

result1_50_50=run_svm_50_50(data1)
#output result to csv files
write.csv(result1_50_50[[3]],"with1_50_50_1.csv",row.names = FALSE)    
write.csv(result1_50_50[[4]],"with1_50_50_2.csv",row.names = FALSE)

result3_50_50=run_svm_50_50(data3)
write.csv(result3_50_50[[3]],"with3_50_50_1.csv",row.names = FALSE)
write.csv(result3_50_50[[4]],"with3_50_50_2.csv",row.names = FALSE)

result1_80_20=run_svm_80_20(data1)
write.csv(result1_80_20[[3]],"with1_80_20.csv",row.names = FALSE)
write.csv(result1_80_20[[4]],"with1_20_80.csv",row.names = FALSE)

result3_80_20=run_svm_80_20(data3)
write.csv(result3_80_20[[3]],"with3_80_20.csv",row.names = FALSE)
write.csv(result3_80_20[[4]],"with3_20_80.csv",row.names = FALSE)
