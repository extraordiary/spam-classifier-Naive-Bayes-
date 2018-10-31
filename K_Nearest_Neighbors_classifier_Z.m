file='spamData.mat';%please change this to the correct path before running it
load (file);
% compute the numbers of features,the data size of test set and train size
Xtraincol=size(Xtrain,2);
ytestrow=size(ytest,1); 
ytrainrow=size(ytrain,1);

% Data processing:  z-normalization
XtrainZ=zscore(Xtrain,1);
XtestZ=zscore(Xtest,1);

% produce the number of samples that should be captured:K
q=1:1:9;
m=10:5:100;
K=[q,m];

% compute the distance between every samples in train set
for i=1:1:ytrainrow
XtrainZ_=repmat(XtrainZ(i,:),ytrainrow,1); %copy the ith sample to every line
Atrian=(XtrainZ_-XtrainZ).^2; % the new matrix minus the old matrix£¬and then squre, for the convience of getting distance in next step
distancetrain(:,i)=sum(Atrian,2).^0.5;
[Sortmatrixtrain(:,i),indtra(:,i)]=sort(distancetrain(:,i));%sort the matrix
end
% compute the distance between every samples in test set
for i=1:1:ytestrow
XtestZ_=repmat(XtestZ(i,:),ytrainrow,1); %copy the ith sample to every line
Atest=(XtestZ_-XtrainZ).^2; % the new matrix minus the old matrix£¬and then squre, for the convience of getting distance in next step
distancetest(:,i)=sum(Atest,2).^0.5;
[Sortmatrixtest(:,i),indtest(:,i)]=sort(distancetest(:,i));%sort the matrix
end
% compute error rate of train set
for n=1:1:size(K,2)
   ktrainZ=0;
  for i=1:1:ytrainrow
    B=indtra(1:K(n),i);
    j=1:1:K(n);
    k=sum(ytrain(B(j)));
    pZtrain(i)=k/K(n);
    resulttra(i)=(pZtrain(i)>0.5);
    if resulttra(i)~=ytrain(i)
       ktrainZ=ktrainZ+1;
    end
  end
  errorrateZ_train(n)=ktrainZ/ytrainrow;
end
% compute errorate of test set
for n=1:1:size(K,2)
   ktestZ=0;
  for i=1:1:ytestrow
    B=indtest(1:K(n),i);
    j=1:1:K(n);
    k=sum(ytrain(B(j)));
    pZtest(i)=k/K(n);
    resulttest(i)=(pZtest(i)>0.5);
    if resulttest(i)~=ytest(i)
       ktestZ=ktestZ+1;
    end
  end
  errorrateZ_test(n)=ktestZ/ytestrow;
end
figure(3);
plot(K,errorrateZ_test,'o-b',K,errorrateZ_train,'o-r')
testerrorZ=[errorrateZ_test(1),errorrateZ_test(10),errorrateZ_test(28)]
trainerrorZ=[errorrateZ_train(1),errorrateZ_train(10),errorrateZ_train(28)]