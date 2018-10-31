file='spamData.mat';%please change this to the correct path before running it
load (file);
% compute the number of features,the data size of test set and train set
Xtraincol=size(Xtrain,2);
ytestrow=size(ytest,1); 
ytrainrow=size(ytrain,1);

% Data processing:  binarization
Xtrainbi=Xtrain;
Xtrainbi((Xtrainbi>0))=1;
Xtrain_bi=[ones(ytrainrow,1),Xtrainbi];
Xtestbi=Xtest;
Xtestbi((Xtestbi>0))=1;
Xtest_bi=[ones(ytestrow,1),Xtestbi];
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
u=sigm(Xtrain_bi*w);
G=transpose(Xtrain_bi)*(u-ytrain);
A=ones(size(u,1),1);
S=diag((A-u).*u);
H=transpose(Xtrain_bi)*S*Xtrain_bi;
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
% compute the error rate of train set after binarization
  p1=sigm(transpose(w)*transpose(Xtrain_bi));
  pbi_tra=(p1>0.5);
  errorratebi_train(n)=size(find(ytrain~=transpose(pbi_tra)),1)/ytrainrow;
% compute the error rate of test set after binarization
  p2=sigm(transpose(w)*transpose(Xtest_bi));
  pbi_test=(p2>0.5);
  errorratebi_test(n)=size(find(ytest~=transpose(pbi_test)),1)/ytestrow;
end
figure(1);
plot(lambda,errorratebi_train,'o-r',lambda,errorratebi_test,'o-b')
testerror_bi=[errorratebi_test(1),errorratebi_test(10),errorratebi_test(28)]
trainerror_bi=[errorratebi_train(1),errorratebi_train(10),errorratebi_train(28)]