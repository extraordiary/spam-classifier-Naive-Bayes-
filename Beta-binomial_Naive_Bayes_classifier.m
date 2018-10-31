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

% class lable=1,the number of features which equals 1 of cth feature,C1(c)
% class lable=0,the number of features which equals 1 of cth feature,C0(c)
XTRAIN=Xtrain;
XTRAIN((XTRAIN>0))=1;% Data processing
A=find(ytrain);% return the element index value of 1 in ytrain
B=find(ytrain==0);% return the element index value of 0 in ytrain
a=size(A);
b=size(B);
for c=1:1:Xtraincol
    d=1:1:a;
    C1(c)=sum(XTRAIN(A(d),c));
    e=1:1:b;
    C0(c)=sum(XTRAIN(B(e),c));
end

% compute the error rate of train set and test set
alpha1=0:0.5:100;
q=1;
for alpha=0:0.5:100 
 ktest=0;
 ktrain=0;
 for j=1:1:ytestrow
      for c=1:1:Xtraincol
        if Xtest(j,c)>0
        ftest(c)=log((C1(c)+alpha)/(N1+2*alpha));
        gtest(c)=log((C0(c)+alpha)/(N0+2*alpha));
        else
        ftest(c)=log(1-(C1(c)+alpha)/(N1+2*alpha));
        gtest(c)=log(1-(C0(c)+alpha)/(N0+2*alpha));
        end
      end
      % predict whether it is spam mail based on Xtest(j,i)
     p1test=log(Py1)+sum(ftest);
     p0test=log(Py0)+sum(gtest);
     p(j)=(p1test>p0test);
     if p(j)~=ytest(j)
     ktest=ktest+1;
     end
 end
  for j=1:1:ytrainrow
      for c=1:1:Xtraincol
        if Xtrain(j,c)>0
        ftrain(c)=log((C1(c)+alpha)/(N1+2*alpha));
        gtrain(c)=log((C0(c)+alpha)/(N0+2*alpha));
        else
        ftrain(c)=log(1-(C1(c)+alpha)/(N1+2*alpha));
        gtrain(c)=log(1-(C0(c)+alpha)/(N0+2*alpha));
        end
      end
      % predict whether it is spam mail based on Xtest(j,i)
     p1train=log(Py1)+sum(ftrain);
     p0train=log(Py0)+sum(gtrain);
     p(j)=(p1train>p0train);
     if p(j)~=ytrain(j)
     ktrain=ktrain+1;
     end
 end
 errorrate_test(q)=ktest/ytestrow;
 errorrate_train(q)=ktrain/ytrainrow;
 q=q+1;
end
plot(alpha1,errorrate_test,alpha1,errorrate_train)
title(' Plots of training and test error rates versus ¦Á')
xlabel(' ¦Á')
ylabel('errorrate')
legend('errorrate-test','errorrate-train')
testerror=[errorrate_test(3),errorrate_test(21),errorrate_test(201)]
trainerror=[errorrate_train(3),errorrate_train(21),errorrate_train(201)]