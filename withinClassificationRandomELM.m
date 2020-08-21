function [ performance ] = withinClassificationRandomELM(data, label, nCross)

indices = crossvalind('KFold', label, nCross);
nChannels = size(data{1},1);
acc = [];

for syn=1:16
    maxSynergies = nChannels- syn + 1;
    for cross = 1:nCross
        trainData{syn,cross} = data{syn}(:,indices ~= cross)';
        testData{syn,cross} = data{syn}(:,indices == cross)';
        trainLabel{syn,cross} = label(indices ~= cross);
        testLabel{syn,cross} = label(indices == cross);

        [trainW0,trainH0] = nnmf(trainData{syn,cross}',maxSynergies,'replicates',100,'algorithm','mult');
        [trainW{syn,cross}, trainH{syn,cross}] = nnmf(trainData{syn,cross}',maxSynergies,'w0',trainW0,'h0',trainH0,'algorithm','als');
        VAF = 1 - sum(sum((trainW{syn,cross}*trainH{syn,cross}).^2))/sum(sum(trainData{syn,cross}'.^2));

        %testW = mrdivide(testData,trainH);
        for columnNumber = 1:size(testData{syn,cross},1)
            testH{syn,cross}(columnNumber,:) = lsqnonneg(trainW{syn,cross},testData{syn,cross}(columnNumber,:)');
        end
        %testW = testData*pinv(trainH);
        
    end
end


totalTime = 0;
ELMcross = 10;
for syn = 1:16
    for cross = 1:nCross
        trainELM = [trainLabel{syn,cross} trainH{syn,cross}'];
        testELM = [testLabel{syn,cross} testH{syn,cross}];
        for elm = 1:ELMcross
            [trainTime, testTime, trainAcc{elm}, testAcc{elm}] = ELM(trainELM, testELM, 1, 150, 'sig');
        end
        acc{syn,cross,1} = trainAcc;
        acc{syn,cross,2} = testAcc;
    end
end

performance = acc;

end

