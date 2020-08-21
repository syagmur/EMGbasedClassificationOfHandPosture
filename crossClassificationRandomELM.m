function [ performance ] = crossClassificationRandom( basedata, data, label, nCross)

indices = crossvalind('KFold', label, nCross);
acc = [];

for syn=1:16
    for cross = 1:nCross

        trainData{syn,cross} = data{syn}(:, indices ~= cross)';
        testData{syn,cross} = data{syn}(:, indices == cross)';
        trainLabel{syn,cross} = label(indices ~= cross,:);
        testLabel{syn,cross} = label(indices == cross,:);

        for columnNumber = 1:size(trainData{syn,cross},1)
            trainH{syn,cross}(columnNumber,:) = lsqnonneg(basedata{syn}', trainData{syn,cross}(columnNumber,:)')';
        end

        for columnNumber = 1:size(testData{syn,cross},1)
            testH{syn,cross}(columnNumber,:) = lsqnonneg(basedata{syn}', testData{syn,cross}(columnNumber,:)')';
        end
    end
end

acc = [];
totalTime = 0;
ELMcross = 10;
for syn = 1:16
    for cross = 1:nCross
        trainELM = [trainLabel{syn,cross} trainH{syn,cross}];
        testELM = [testLabel{syn,cross} testH{syn,cross}];
        parfor elm = 1:ELMcross
            [trainTime, testTime, trainAcc{elm}, testAcc{elm}] = ELM(trainELM, testELM, 1, 150, 'sig');
        end
        acc{syn,cross,1} = trainAcc;
        acc{syn,cross,2} = testAcc;
    end
end

% sprintf('Total testing time is %f.',totalTime)

performance = acc;
end

