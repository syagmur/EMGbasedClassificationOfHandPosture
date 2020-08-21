function [ performance ] = withinClassificationELM(data, label, nCross, cond)
% If cond is 1, generate a random base matrix
indices = crossvalind('KFold', label, nCross);
nChannels = size(data,1);
data = data';
acc = [];

for syn=1:16
    maxSynergies = nChannels- syn + 1;
    for cross = 1:nCross

        trainData{syn,cross} = data(indices ~= cross,:)';
        testData{syn,cross} = data(indices == cross,:);
        trainLabel{syn,cross} = label(indices ~= cross,:)';
        testLabel{syn,cross} = label(indices == cross,:);

        
        [trainW0,trainH0] = nnmf(trainData{syn,cross}, maxSynergies,'replicates',100,'algorithm','mult');
        [trainW{syn,cross},trainH{syn,cross}] = nnmf(trainData{syn,cross},maxSynergies,'w0',trainW0,'h0',trainH0,'algorithm','als');
        VAF{syn} = 1 - sum(sum((trainW{syn,cross}*trainH{syn,cross}).^2))/sum(sum(trainData{syn,cross}.^2));
        
        if cond==1
            mu = mean(trainW{syn,cross}');
            sigma  = var(trainW{syn,cross});
            trainW{syn,cross} = abs(repmat(mu',[1,maxSynergies]) + (randn(size(trainW{syn,cross},1), size(trainW{syn,cross},2)).*sigma));           
        elseif cond ==2
            trainW{syn,cross} = trainW{syn,cross}(randperm(size(trainW{syn,cross},1)),:);
        end
        
        %testW = mrdivide(testData,trainH);
        for columnNumber = 1:size(testData{syn,cross},1)
            testH{syn,cross}(columnNumber,:) = lsqnonneg(trainW{syn,cross},testData{syn,cross}(columnNumber,:)')';
        end
        
    end
end


% parfor syn = 1:15
%     for cross = 1:nCross
%         svmStruct = templateSVM('Standardize',1,'KernelFunction','gaussian');
%         svmModel = fitcecoc(trainH{syn,cross}', trainLabel{syn,cross}, 'Learners',svmStruct);
%         resultLabel = predict(svmModel, testH{syn,cross});
%         perf = classperf(testLabel{syn,cross},resultLabel);
%         acc(syn,cross) = perf.CorrectRate;
%     end
% end
% 
% performance = mean(acc,2);

totalTime = 0;
ELMcross = 10;
for syn = 1:16
    for cross = 1:nCross
        trainELM = [trainLabel{syn,cross}' trainH{syn,cross}'];
        testELM = [testLabel{syn,cross} testH{syn,cross}];
        parfor elm = 1:ELMcross
            [trainTime, testTime, trainAcc{elm}, testAcc{elm}] = ELM(trainELM, testELM, 1, 200, 'sig');
        end
        acc{syn,cross,1} = trainAcc;
        acc{syn,cross,2} = testAcc;
    end
end

performance = acc;

end

