classifier3bi
classifier3log
classifier3Z
figure(4)
plot(lambda,errorratebi_train,'r',lambda,errorratebi_test,'--r',lambda,errorratelog_train,'k',lambda,errorratelog_test,'--k',lambda,errorrateZ_train,'b',lambda,errorrateZ_test,'--b')
title(' Plots of training and test error rates versus ¦Ë')
xlabel('¦Ë')
ylabel('errorrate')
legend('errorratebi-train','errorratebi-test','errorratelog-train','errorratelog-test','errorrateZ-train','errorrateZ-test')