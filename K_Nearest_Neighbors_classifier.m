classifier4bi
classifier4log
classifier4Z
figure(4)
plot(K,errorratebi_train,'r',K,errorratebi_test,'--r',K,errorratelog_train,'k',K,errorratelog_test,'--k',K,errorrateZ_train,'b',K,errorrateZ_test,'--b')
title(' Plots of training and test error rates versus K')
xlabel('K')
ylabel('errorrate')
legend('errorratebi-train','errorratebi-test','errorratelog-train','errorratelog-test','errorrateZ-train','errorrateZ-test')