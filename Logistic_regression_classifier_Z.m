file='spamData.mat';%please change this to the correct path before running it
load (file);
% compute the number of features,the data size of test set and train size
Xtraincol=size(Xtrain,2);
ytestrow=size(ytest,1); 
ytrainrow=size(ytrain,1);

% Data processing:  binarization
XtrainZ=zscore(Xtrain,1);
Xtrain_Z=[ones(ytrainrow,1),XtrainZ];
XtestZ=zscore(Xtest,1);
Xtest_Z=[ones(ytestrow,1),XtestZ];
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
u=sigm(Xtrain_Z*w);
G=transpose(Xtrain_Z)*(u-ytrain);
A=ones(size(u,1),1);
S=diag((A-u).*u);
H=transpose(Xtrain_Z)*S*Xtrain_Z;
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
% compute the error rate of train set after Z-normalization
  p1=sigm(transpose(w)*transpose(Xtrain_Z));
  plZ_tra=(p1>0.5);
  errorrateZ_train(n)=size(find(ytrain~=transpose(plZ_tra)),1)/ytrainrow;
% compute the error rate of test set after Z-normalization
  p2=sigm(transpose(w)*transpose(Xtest_Z));
  plZ_test=(p2>0.5);
  errorrateZ_test(n)=size(find(ytest~=transpose(plZ_test)),1)/ytestrow;
end
figure(3);
plot(lambda,errorrateZ_train,'o-r',lambda,errorrateZ_test,'o-b')
testerror_Z=[errorrateZ_test(1),errorrateZ_test(10),errorrateZ_test(28)]
trainerror_Z=[errorrateZ_train(1),errorrateZ_train(10),errorrateZ_train(28)]