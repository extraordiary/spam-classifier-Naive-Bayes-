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

% Data processing:log-transform of the train set and test set,get trainlog and testlog
trainlog=log(Xtrain+0.1);
testlog=log(Xtest+0.1);

% compute the mean and the variance of cth feature when class label is 1 in train set 
% compute the mean and the variance of cth feature when class label is 0 in train set
A=find(ytrain);% return the element index value of 1 in ytrain
B=find(ytrain==0);% return the element index value of 0 in ytrain
a=size(A,1);
b=size(B,1);
for k=1:1:a
    label1(k,:)=trainlog(A(k),:);
end
for j=1:1:b
    label0(j,:)=trainlog(B(j),:);
end
train_mean1=mean(label1);
train_mean0=mean(label0);
train_sigma1=std(label1,1);
train_sigma0=std(label0,1);
train_sigma1=train_sigma1+1e-3;
train_sigma0=train_sigma0+1e-3;
%compute the error rate of train set after log-transform
for i=1:1:ytrainrow
    multiplelog1train(i,:)=log(normpdf(trainlog(i,:),train_mean1,train_sigma1));
    multiplelog0train(i,:)=log(normpdf(trainlog(i,:),train_mean0,train_sigma0));
end
P1_train=sum(multiplelog1train,2)+log(Py1);
P0_train=sum(multiplelog0train,2)+log(Py0);
erroratelog_train=size(find(ytrain~=(P1_train>P0_train)),1)/ytrainrow
%compute the error rate of test set after log-transform
for i=1:1:ytestrow
    multiplelog1test(i,:)=log(normpdf(testlog(i,:),train_mean1,train_sigma1));
    multiplelog0test(i,:)=log(normpdf(testlog(i,:),train_mean0,train_sigma0)); 
end
P1_test=sum(multiplelog1test,2)+log(Py1);
P0_test=sum(multiplelog0test,2)+log(Py0);
erroratelog_test=size(find(ytest~=(P1_test>P0_test)),1)/ytestrow