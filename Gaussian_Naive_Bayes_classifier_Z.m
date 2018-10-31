file='spamData.mat';%please change this to the correct path before running it
load (file);
% compute the number of features,the data size of test set and train size
Xtraincol=size(Xtrain,2);
ytestrow=size(ytest,1); 
ytrainrow=size(ytrain,1);

% compute the prior ------  Py0 and Py1,when y=0 and y=1 respectively
N1=sum(ytrain,1);
Py1=N1/ytrainrow;
N0=ytrainrow-N1;
Py0=N0/ytrainrow;

% Data processing:Z-normalization of the train set and test set,get trainZ and testZ
trainZ=zscore(Xtrain,1);
testZ=zscore(Xtest,1);

% compute the mean and the variance of cth feature when class label is 1 in train set 
% compute the mean and the variance of cth feature when class label is 0 in train set
A=find(ytrain);% return the element index value of 1 in ytrain
B=find(ytrain==0);% return the element index value of 0 in ytrain
a=size(A,1);
b=size(B,1);
for k=1:1:a
    label1(k,:)=trainZ(A(k),:);
end
for j=1:1:b
    label0(j,:)=trainZ(B(j),:);
end
train_mean1=mean(label1);
train_mean0=mean(label0);
train_sigma1=std(label1,1);
train_sigma0=std(label0,1);
train_sigma1=train_sigma1+1e-3;
train_sigma0=train_sigma0+1e-3;
%compute the error rate of train set after Z-normalization
for i=1:1:ytrainrow
    multipleZ1train(i,:)=log(normpdf(trainZ(i,:),train_mean1,train_sigma1));
    multipleZ0train(i,:)=log(normpdf(trainZ(i,:),train_mean0,train_sigma0));
end
P1_train=sum(multipleZ1train,2)+log(Py1);
P0_train=sum(multipleZ0train,2)+log(Py0);
errorateZ_train=size(find(ytrain~=(P1_train>P0_train)),1)/ytrainrow
%compute the error rate of test set after Z-normalization
for i=1:1:ytestrow
  multipleZ1test(i,:)=log(normpdf(testZ(i,:),train_mean1,train_sigma1));
  multipleZ0test(i,:)=log(normpdf(testZ(i,:),train_mean0,train_sigma0));
end
P1_test=sum(multipleZ1test,2)+log(Py1);
P0_test=sum(multipleZ0test,2)+log(Py0);
errorateZ_test=size(find(ytest~=(P1_test>P0_test)),1)/ytestrow