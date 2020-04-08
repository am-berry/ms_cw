function [acc, recall, spec, prec, f1, fmi] = evaluation(ytest, ypred)

TP = 0;
FP = 0;
TN = 0;
FN = 0;

for i = 1:length(ytest)
    if ypred(i) == 1
        if ytest(i) == 1
            TP = TP + 1;
        elseif ytest(i) == 0
            FP = FP +1;
        end
    elseif ypred(i) == 0
        if ytest(i) == 1
            FN = FN + 1;
        elseif ytest(i) == 0
            TN = TN + 1;
        end
    end
end

acc = (TP+TN) / (TP+TN+FP+FN);
recall = TP/(TP+FN);
spec = TN/(TN+FP);
prec = TP/(TP+FP);
f1 = (2*prec*recall) / (prec+recall);
fmi = sqrt(prec*recall);

end

            