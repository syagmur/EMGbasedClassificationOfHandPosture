%%
% Random wrt activation
fprintf('\nRandom ASL to Proned');
perfASLtoPronedRandom = crossClassificationRandomELM(baseASL, randPronedActivation, labelProned, 10);
fprintf('\nRandom SemiProned to Proned')
perfSemiPronedtoPronedRandom = crossClassificationRandomELM(baseSemiProned, randPronedActivation, labelProned, 10);
fprintf('\nRandom Grasp to Proned')
perfGrasptoPronedRandom = crossClassificationRandomELM(baseGrasp, randPronedActivation, labelProned, 10);
fprintf('\nRandom Natural to Proned')
perfNaturaltoPronedRandom = crossClassificationRandomELM(baseNatural, randPronedActivation, labelProned, 10);
fprintf('\nRandom Proned to Proned\n')
perfPronedtoPronedRandom = withinClassificationRandomELM(randPronedActivation, labelProned, 10);

% Random wrt activation
fprintf('\nRandom Proned to ASL')
perfPronedtoASLRandom = crossClassificationRandomELM(baseProned, randASLActivation, labelASL, 10);
fprintf('\nRandom SemiProned to ASL')
perfSemiPronedtoASLRandom = crossClassificationRandomELM(baseSemiProned, randASLActivation, labelASL, 10);
fprintf('\nRandom Natural to ASL')
perfNaturaltoASLRandom = crossClassificationRandomELM(baseNatural, randASLActivation, labelASL, 10);
fprintf('\nRandom ASL to ASL\n')
perfASLtoASLRandom = withinClassificationRandomELM(randASLActivation, labelASL, 10);
fprintf('\nRandom Grasp to ASL')
perfGrasptoASLRandom = crossClassificationRandomELM(baseGrasp, randASLActivation, labelASL, 10);
% Random wrt activation
fprintf('\nRandom ASL to SemiProned')
perfASLtoSemiPronedRandom  = crossClassificationRandomELM(baseASL, randSemiPronedActivation, labelSemiProned, 10);
fprintf('\nRandom Proned to SemiProned')
perfPronedtoSemiPronedRandom  = crossClassificationRandomELM(baseProned, randSemiPronedActivation, labelSemiProned, 10);
fprintf('\nRandom Grasp to SemiProned')
perfGrasptoSemiPronedRandom  = crossClassificationRandomELM(baseGrasp, randSemiPronedActivation, labelSemiProned, 10);
fprintf('\nRandom Natural to SemiProned')
perfNaturaltoSemiPronedRandom  = crossClassificationRandomELM(baseNatural, randSemiPronedActivation, labelSemiProned, 10);
fprintf('\nRandom SemiProned to SemiProned\n')
perfSemiPronedtoSemiPronedRandom  = withinClassificationRandomELM(randSemiPronedActivation, labelSemiProned, 10);
% Random wrt activation
fprintf('\nRandom ASL to Grasp')
perfASLtoGraspRandom = crossClassificationRandomELM(baseASL, randGraspActivation, labelGrasp, 10);
fprintf('\nRandom Proned to Grasp')
perfPronedtoGraspRandom = crossClassificationRandomELM(baseProned, randGraspActivation, labelGrasp, 10);
fprintf('\nRandom SemiProned to Grasp')
perfSemiPronedtoGraspRandom = crossClassificationRandomELM(baseSemiProned, randGraspActivation, labelGrasp, 10);
fprintf('\nRandom Natural to Grasp')
perfNaturaltoGraspRandom = crossClassificationRandomELM(baseNatural, randGraspActivation, labelGrasp, 10);
fprintf('\nRandom Grasp to Grasp\n')
perfGrasptoGraspRandom = withinClassificationRandomELM(randGraspActivation, labelGrasp, 10);

performanceMatrixRandomActivation = [perfASLtoGraspRandom,perfPronedtoGraspRandom,perfSemiPronedtoGraspRandom,perfNaturaltoGraspRandom,perfGrasptoGraspRandom,...
    perfASLtoASLRandom,perfPronedtoASLRandom,perfSemiPronedtoASLRandom,perfNaturaltoASLRandom, perfGrasptoASLRandom,...
    perfASLtoPronedRandom,perfPronedtoPronedRandom,perfSemiPronedtoPronedRandom,perfNaturaltoPronedRandom,perfGrasptoPronedRandom,...
    perfASLtoSemiPronedRandom,perfPronedtoSemiPronedRandom,perfSemiPronedtoSemiPronedRandom,perfNaturaltoSemiPronedRandom,perfGrasptoSemiPronedRandom];
save(fullfile(path2, 'ELMperformanceMatrixRandomActivation2.mat'), 'performanceMatrixRandomActivation', '-mat');
%