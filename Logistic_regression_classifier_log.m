file='spamData.mat';%please change this to the correct path before running it
load (file);
% compute the number of features,the data size of test set and train set
Xtraincol=size(Xtrain,2);
ytestrow=size(ytest,1); 
ytrainrow=size(ytrain,1);

% Data processing:  log-transform
Xtrainlog=log(Xtrain+0.1);
Xtrain_log=[ones(ytrainrow,1),Xtrainlog];
Xtestlog=log(Xtest+0.1);
Xtest_log=[ones(ytestrow,1),Xtestlog];
% difine a function sigm
sigm=inline('1./(1+exp(-x))','x');

% use Newton¡¯s Method for Logistic Regression to compute w
unitmatrix_change=ones(1,Xtraincol+1);
unitmatrix_change(1,1)=0;
unitmatrix_change=diag(unitmatrix_change);
q=1:1:9;
m=10:5:100;
lambda=[q,m];
for n=1:1:size(lambda,2)
a=1;
w=zeros(Xtraincol+1,1);
while a~=0
u=sigm(Xtrain_log*w);
G=transpose(Xtrain_log)*(u-ytrain);
A=ones(size(u,1),1);
S=diag((A-u).*u);
H=transpose(Xtrain_log)*S*Xtrain_log;
w1=w;
w1(1,:)=0;
Greg=G+lambda(n)*w1;% Exclude Bias from l2 Regularization
Hreg=H+lambda(n)*unitmatrix_change;% Exclude Bias from l2 Regularization
w=w-pinv(Hreg)*Greg;
d=-pinv(Hreg)*Greg;
if sum(d.^2)<1e-5
    a=0;
end
end
% compute the error rate of train set after log-transform
  p1=sigm(transpose(w)*transpose(Xtrain_log));
  plog_tra=(p1>0.5);
  errorratelog_train(n)=size(find(ytrain~=transpose(plog_tra)),1)/ytrainrow;
% compute the error rate of test set after log-transform
  p2=sigm(transpose(w)*transpose(Xtest_log));
  plog_test=(p2>0.5);
  errorratelog_test(n)=size(find(ytest~=transpose(plog_test)),1)/ytestrow;
end
figure(2);
plot(lambda,errorratelog_train,'o-r',lambda,errorratelog_test,'o-b')
testerror_log=[errorratelog_test(1),errorratelog_test(10),errorratelog_test(28)]
trainerror_log=[errorratelog_train(1),errorratelog_train(10),errorratelog_train(28)]