file='spamData.mat';%please change this to the correct path before running it
load (file);
% compute the numbers of features,the data size of test set and train size
Xtraincol=size(Xtrain,2);
ytestrow=size(ytest,1); 
ytrainrow=size(ytrain,1);

% Data processing:  log-transform
Xtrainlog=log(Xtrain+0.1);
Xtestlog=log(Xtest+0.1);

% produce the number of samples that should be captured:K
q=1:1:9;
m=10:5:100;
K=[q,m];

% compute the distance between every samples in train set
for i=1:1:ytrainrow
Xtrainlog_=repmat(Xtrainlog(i,:),ytrainrow,1); %copy the ith sample to every line
Atrian=(Xtrainlog_-Xtrainlog).^2; % the new matrix minus the old matrix£¬and then squre, for the convience of getting distance in next step
distancetrain(:,i)=sum(Atrian,2).^0.5;
[Sortmatrixtrain(:,i),indtra(:,i)]=sort(distancetrain(:,i));%sort the matrix
end
% compute the distance between every samples in test set
for i=1:1:ytestrow
Xtestlog_=repmat(Xtestlog(i,:),ytrainrow,1); %copy the ith sample to every line
Atest=(Xtestlog_-Xtrainlog).^2; % the new matrix minus the old matrix£¬and then squre, for the convience of getting distance in next step
distancetest(:,i)=sum(Atest,2).^0.5;
[Sortmatrixtest(:,i),indtest(:,i)]=sort(distancetest(:,i));%sort the matrix
end
% compute errorate of train set
for n=1:1:size(K,2)
   ktrainl=0;
  for i=1:1:ytrainrow
    B=indtra(1:K(n),i);
    j=1:1:K(n);
    k=sum(ytrain(B(j)));
    plogtrain(i)=k/K(n);
    resulttra(i)=(plogtrain(i)>0.5);
    if resulttra(i)~=ytrain(i)
       ktrainl=ktrainl+1;
    end
  end
  errorratelog_train(n)=ktrainl/ytrainrow;
end
% compute error rate of test set
for n=1:1:size(K,2)
   ktestl=0;
  for i=1:1:ytestrow
    B=indtest(1:K(n),i);
    j=1:1:K(n);
    k=sum(ytrain(B(j)));
    plogtest(i)=k/K(n);
    resulttest(i)=(plogtest(i)>0.5);
    if resulttest(i)~=ytest(i)
       ktestl=ktestl+1;
    end
  end
  errorratelog_test(n)=ktestl/ytestrow;
end
figure(2);
plot(K,errorratelog_test,'o-b',K,errorratelog_train,'o-r')
testerrorlog=[errorratelog_test(1),errorratelog_test(10),errorratelog_test(28)]
trainerrorlog=[errorratelog_train(1),errorratelog_train(10),errorratelog_train(28)]