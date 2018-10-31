file='spamData.mat';%please change this to the correct path before running it
load (file);
% compute the number of features,the data size of test set and train size
Xtraincol=size(Xtrain,2);
ytestrow=size(ytest,1); 
ytrainrow=size(ytrain,1);

% Data processing: binarization
Xtrainbi=Xtrain;
Xtrainbi((Xtrainbi>0))=1;
Xtestbi=Xtest;
Xtestbi((Xtestbi>0))=1;

% produce the number of samples that should be captured:K
q=1:1:9;
m=10:5:100;
K=[q,m];

% compute the distance between every samples in train set
for i=1:1:ytrainrow
Xtrainbi_=repmat(Xtrainbi(i,:),ytrainrow,1);%copy the ith sample to every line
Atrian=(Xtrainbi_-Xtrainbi).^2;%the new matrix minus the old matrix£¬and then squre, for the convience of getting distance in next step
distancetrain(:,i)=sum(Atrian,2).^0.5;
[Sortmatrixtrain(:,i),indtra(:,i)]=sort(distancetrain(:,i));%sort the matrix
end
% compute the distance between every samples in test set
for i=1:1:ytestrow
Xtestbi_=repmat(Xtestbi(i,:),ytrainrow,1);%copy the ith sample to every line
Atest=(Xtestbi_-Xtrainbi).^2;%the new matrix minus the old matrix£¬and then squre, for the convience of getting distance in next step
distancetest(:,i)=sum(Atest,2).^0.5;
[Sortmatrixtest(:,i),indtest(:,i)]=sort(distancetest(:,i));%sort the matrix
end

% compute errorate of train set
for n=1:1:size(K,2)
   ktrainbi=0;
  for i=1:1:ytrainrow
    B=indtra(1:K(n),i);
    j=1:1:K(n);
    k=sum(ytrain(B(j)));
    pbitrain(i)=k/K(n);
    resulttra(i)=(pbitrain(i)>0.5);
    if resulttra(i)~=ytrain(i)
       ktrainbi=ktrainbi+1;
    end
  end
  errorratebi_train(n)=ktrainbi/ytrainrow;
end
% compute error rate of test set
for n=1:1:size(K,2)
   ktestbi=0;
  for i=1:1:ytestrow
    B=indtest(1:K(n),i);
    j=1:1:K(n);
    k=sum(ytrain(B(j)));
    pbitest(i)=k/K(n);
    resulttest(i)=(pbitest(i)>0.5);
    if resulttest(i)~=ytest(i)
       ktestbi=ktestbi+1;
    end
  end
  errorratebi_test(n)=ktestbi/ytestrow;
end
figure(1);
plot(K,errorratebi_test,'o-b',K,errorratebi_train,'o-r')
testerrorbi=[errorratebi_test(1),errorratebi_test(10),errorratebi_test(28)]
trainerrorbi=[errorratebi_train(1),errorratebi_train(10),errorratebi_train(28)]