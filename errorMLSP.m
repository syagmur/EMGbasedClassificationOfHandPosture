% Data = Data';
% perfASLtoGrasp = Data(:,1);
% perfPronedtoGrasp = Data(:,2);
% perfSemiPronedtoGrasp = Data(:,3);
% perfNaturaltoGrasp = Data(:,4);
% perfGrasptoGrasp = Data(:,5);
% perfASLtoASL = Data(:,6);
% perfPronedtoASL = Data(:,7);
% perfSemiPronedtoASL = Data(:,8);
% perfNaturaltoASL = Data(:,9);
% perfGrasptoASL = Data(:,10);
% perfASLtoProned = Data(:,11);
% perfPronedtoProned = Data(:,12);
% perfSemiPronedtoProned = Data(:,13);
% perfNaturaltoProned = Data(:,14);
% perfGrasptoProned = Data(:,15);
% perfASLtoSemiProned = Data(:,16);
% perfPronedtoSemiProned = Data(:,17);
% perfSemiPronedtoSemiProned = Data(:,18);
% perfNaturaltoSemiProned = Data(:,19);
% perfGrasptoSemiProned = Data(:,20);
% 
% 
% 
% prList = [9,7,6,14,12,11]; 
% subPerf{i} = Data(prList,:);
% pe = [];
% for i=1:5
%    pe(i,:,:) = subPerf{i};   
% end

% av = squeeze(mean(pe,1));
% va = squeeze(var(pe,1));

subplot(2,1,1);
clear all
svmAve = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\SVMAve');
svmAve = svmAve.av;
svmVar = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\SVMva');
svmVar = svmVar.va;
elmAve = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmAve');
elmAve = elmAve.av;
elmVar = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmVa');
elmVar = elmVar.va;
svmAveRandAct = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\SVMAveRandAct');
svmAveRandAct = svmAveRandAct.av;
svmVarRandAct = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\SVMvarRandAct');
svmVarRandAct = svmVarRandAct.va;
svmAveRandBase = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\SVMAveRandBase');
svmAveRandBase = svmAveRandBase.av;
svmVarRandBase = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\SVMVarRandBase');
svmVarRandBase = svmVarRandBase.va;
elmSubAve1 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub1Ave');
elmSubAve1 = elmSubAve1.meanNew;
elmSubVar1 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub1Var');
elmSubVar1 = elmSubVar1.varNew;
elmSubAve2 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub2Ave');
elmSubAve2 = elmSubAve2.meanNew;
elmSubVar2 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub2Var');
elmSubVar2 = elmSubVar2.varNew;
elmSubAve3 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub3Ave');
elmSubAve3 = elmSubAve3.meanNew;
elmSubVar4 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub4Var');
elmSubVar4 = elmSubVar4.varNew;
elmSubAve4 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub4Ave');
elmSubAve4 = elmSubAve4.meanNew;
elmSubVar3 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub5Var');
elmSubVar3 = elmSubVar3.varNew;
elmSubAve5 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub5Ave');
elmSubAve5 = elmSubAve5.meanNew;
elmSubVar5 = load('S:\CSL\2018\_Rejected\C_Gunay_MuscleSynergiesGeneralizability_MLSP_2018\Results\elmSub3Var');
elmSubVar5 = elmSubVar5.varNew;

for i=1:6
    svmBor = [svmMin(i,:); svmMax(i,:)];
    elmBor = [elmMin(i,:); elmMax(i,:)];
    minVal = [elmMin(i,:); svmMin(i,:)];
    maxVal = [elmMax(i,:)-elmAve(i,:); svmMax(i,:)-elmAve(i,:)];    
    res = [elmAve(i,:); svmAve(i,:)];
    errorbar_groups(res, minVal, maxVal);
end

elmSubFoldAllAve(1,:,:) = squeeze(mean(elmSubAve1,3));
elmSubFoldAllAve(2,:,:) = squeeze(mean(elmSubAve2,3));
elmSubFoldAllAve(3,:,:) = squeeze(mean(elmSubAve3,3));
elmSubFoldAllAve(4,:,:) = squeeze(mean(elmSubAve4,3));
elmSubFoldAllAve(5,:,:) = squeeze(mean(elmSubAve5,3));

elmAveAll = squeeze(mean(elmSubFoldAllAve,1));


elmSubFoldAllVar(1,:,:) = squeeze(mean(elmSubVar1,3));
elmSubFoldAllVar(2,:,:) = squeeze(mean(elmSubVar2,3));
elmSubFoldAllVar(3,:,:) = squeeze(mean(elmSubVar3,3));
elmSubFoldAllVar(4,:,:) = squeeze(mean(elmSubVar4,3));
elmSubFoldAllVar(5,:,:) = squeeze(mean(elmSubVar5,3));

elmVarAll = squeeze(mean(elmSubFoldAllVar,1));


%% 
% SVM errorbar plots - MLSP
figure;
subplot(1,2,1);
errorbar(16:-1:1, svmAve(1,:), svmVar(1,:),'r-*'); hold on;
errorbar(16:-1:1, svmAve(2,:), svmVar(2,:),'g-*'); hold on;
errorbar(16:-1:1, svmAve(3,:), svmVar(3,:),'b-*'); hold on;
plot(16:-1:1, svmAveRandAct(1,:), 'r:',16:-1:1, svmAveRandAct(2,:), 'g:',16:-1:1,...
    svmAveRandAct(3,:), 'b:', 'LineWidth',1.5); hold on;
plot(16:-1:1, svmAveRandBase(1,:), 'r--',16:-1:1, svmAveRandBase(2,:), 'g--',16:-1:1,...
    svmAveRandBase(3,:), 'b--', 'LineWidth',1.5) 
xlabel('Number of Synergies')
ylabel('Performance')
legend('FREE','Grasp', 'ASL','FREE - RA','Grasp - RA', 'ASL - RA', 'FREE - RB','Grasp - RB', 'ASL - RB')
title('ASL Classification','interpreter','latex')

subplot(1,2,2);
h(1) = errorbar(16:-1:1, svmAve(4,:), svmVar(4,:),'r-*'); hold on;
h(2) = errorbar(16:-1:1, svmAve(5,:), svmVar(5,:),'g-*'); hold on;
h(3) = errorbar(16:-1:1, svmAve(6,:), svmVar(6,:),'b-*'); hold on;
t = plot(15:-1:1, svmAveRandAct(4,:), 'r:',15:-1:1, svmAveRandAct(5,:), 'g:',15:-1:1,...
    svmAveRandAct(6,:), 'b:', 'LineWidth',1.5); hold on;
k = plot(15:-1:1, svmAveRandBase(4,:), 'r--',15:-1:1, svmAveRandBase(5,:), 'g--',15:-1:1,...
    svmAveRandBase(6,:), 'b--', 'LineWidth',1.5) ;
xlabel('Number of Synergies')
ylabel('Performance')
%legend([h(1) t(1) k(1)],{'Actual Results','Random Activation', 'Random Base'})
title('Grasp Mimicking Classification','interpreter','latex')

%%


%%
% SVM vs ELM
% NER Plot 2
figure;
subplot(1,2,1);
plot(1:15, SVMMean(1,:), 'r-o', 'LineWidth',1.5); hold on;
plot(1:15, SVMMean(2,:),'-o','color',[0, 0.5, 0], 'LineWidth',1.5)
plot(1:15, SVMMean(3,:), 'b-o', 'LineWidth',1.5)
plot(1:15, elmAve(1,:), 'r--x', 'LineWidth',1.5)
plot(1:15, elmAve(2,:),'--x','color',[0, 0.5, 0], 'LineWidth',1.5)
plot(1:15, elmAve(3,:), 'b--x', 'LineWidth',1.5) 
xlabel('Number of Synergies','interpreter','latex','FontSize',18,'FontWeight','bold')
ylabel('Performance','interpreter','latex','FontSize',18,'FontWeight','bold')
legend('Free - SVM','Grasp - SVM', 'ASL - SVM','Free - ELM','Grasp - ELM', 'ASL - ELM','interpreter','latex')
title('ASL Classification', 'interpreter','latex','FontSize',18,'FontWeight','bold')
ylim([0 1])
xlim([0 16])
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]);
grid on;

subplot(1,2,2);
plot(1:15, SVMMean(4,:), 'r-o'); hold on;
plot(1:15, SVMMean(5,:), '-o','color',[0, 0.5, 0], 'LineWidth',1.5)
plot(1:15, SVMMean(6,:), 'b-o', 'LineWidth',1.5)
plot(1:15, elmAve(4,:), 'r--x', 'LineWidth',1.5)
plot(1:15, elmAve(5,:),'--x','color',[0, 0.5, 0], 'LineWidth',1.5)
plot(1:15, elmAve(6,:), 'b--x', 'LineWidth',1.5) 
xlabel('Number of Synergies','interpreter','latex','FontSize',18,'FontWeight','bold')
ylabel('Performance','interpreter','latex','FontSize',18,'FontWeight','bold')
% legend('Free Movement','Grasp', 'ASL')
ylim([0 1])
xlim([0 16])
title('Grasp Mimicking Classification','interpreter','latex','FontSize',18,'FontWeight','bold')
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]);
grid on;
%%

p=randperm(10);
figure;
subplot(1,2,1);
errorbar(16:-1:1, elmSubAve(1:16,1,p(1)), elmSubVar(16:-1:1,1,p(1)),'r-'); hold on;
errorbar(16:-1:1, elmSubAve(1:16,2,p(1)), elmSubVar(16:-1:1,2,p(1)),'g-'); hold on;
errorbar(16:-1:1, elmSubAve(1:16,3,p(1)), elmSubVar(16:-1:1,3,p(1)),'b-'); hold on;
xlabel('Number of Synergies','interpreter','latex','FontSize',18,'FontWeight','bold')
ylabel('Performance','interpreter','latex')
legend('Free Movement','Grasp', 'ASL','interpreter','latex')
title('ASL Classification','interpreter','latex')
subplot(1,2,2);
h(1) = errorbar(16:-1:1, elmSubAve(1:16,4,p(1)), elmSubVar(16:-1:1,4,p(1)),'r-'); hold on;
h(2) = errorbar(16:-1:1, elmSubAve(1:16,5,p(1)), elmSubVar(16:-1:1,5,p(1)),'g-'); hold on;
h(3) = errorbar(16:-1:1, elmSubAve(1:16,6,p(1)), elmSubVar(16:-1:1,6,p(1)),'b-'); hold on;
xlabel('Number of Synergies','interpreter','latex')
ylabel('Performance','interpreter','latex')
% legend('Free Movement','Grasp', 'ASL')
title('Grasp Mimicking Classification','interpreter','latex')


subplot(1,2,1);
% plot(15:-1:1, elmSubVar(15:-1:1,1,p(1)), 'r-', 15:-1:1, elmSubVar(15:-1:1,2,p(1)), 'g-', 15:-1:1, elmSubVar(15:-1:1,3,p(1)), 'b-')
bar(15:-1:1, [elmSubVar(15:-1:1,1,p(1)) elmSubVar(15:-1:1,2,p(1)) elmSubVar(15:-1:1,3,p(1))]); hold on;
% bar(15:-1:1, elmSubVar(15:-1:1,2,p(1)), 'g-'); hold on;
% bar(15:-1:1, elmSubVar(15:-1:1,3,p(1)), 'b-'); hold on;
xlabel('Number of Synergies')
ylabel('ELM Variation')
legend('FREE','Grasp', 'ASL')
title('ASL Classification')
subplot(1,2,2);
% plot(15:-1:1, elmSubVar(15:-1:1,4,p(1)), 'r-',15:-1:1, elmSubVar(15:-1:1,5,p(1)), 'g-',15:-1:1, elmSubVar(15:-1:1,6,p(1)), 'b-')
% bar(15:-1:1, elmSubVar(15:-1:1,4,p(1)), 'r-'); hold on;
% bar(15:-1:1, elmSubVar(15:-1:1,5,p(1)), 'g-'); hold on;
% bar(15:-1:1, elmSubVar(15:-1:1,6,p(1)), 'b-'); hold on;
bar(15:-1:1, [elmSubVar(15:-1:1,4,p(1)) elmSubVar(15:-1:1,5,p(1)) elmSubVar(15:-1:1,6,p(1))]); hold on;

xlabel('Number of Synergies')
ylabel('ELM Variation')
legend('FREE','Grasp', 'ASL')
% legend('Free Movement','Grasp', 'ASL')
title('Grasp Mimicking Classification')



SVMMean = SVMResults.MeanGroupData([9,7,6,14,12,11],1:15);
SVMSTD = SVMResults.StdGroupData([9,7,6,14,12,11],1:15)./(5^1/2);
SVMMeanRB = SVMResults.MeanGroupDataBase([9,7,6,14,12,11],1:15);
SVMSTDRB = SVMResults.StdGroupDataBase([9,7,6,14,12,11],1:15)./(5^1/2);
SVMMeanRA = SVMResults.MeanGroupDataAct([9,7,6,14,12,11],1:15);
SVMSTDRA = SVMResults.StdGroupDataAct([9,7,6,14,12,11],1:15)./(5^1/2);

% NER Plot 1
figure;
subplot(1,2,1);
errorbar(1:15, SVMMean(1,:), SVMSTD(1,:),'r-*'); hold on;
errorbar(1:15, SVMMean(2,:), SVMSTD(2,:),'-*','color',[0, 0.5, 0]); hold on;
errorbar(1:15, SVMMean(3,:), SVMSTD(3,:),'b-*'); hold on;
errorbar(1:15, SVMMeanRA(1,:), SVMSTDRA(1,:),'r:'); hold on;
errorbar(1:15, SVMMeanRA(2,:), SVMSTDRA(2,:),':','color',[0, 0.5, 0]); hold on;
errorbar(1:15, SVMMeanRA(3,:), SVMSTDRA(3,:),'b:'); hold on;
errorbar(1:15, SVMMeanRB(1,:), SVMSTDRB(1,:),'r--'); hold on;
errorbar(1:15, SVMMeanRB(2,:), SVMSTDRB(2,:),'--','color',[0, 0.5, 0]); hold on;
errorbar(1:15, SVMMeanRB(3,:), SVMSTDRB(3,:),'b--'); hold on;
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]);
ylim([0 1])
xlim([0 16])
% plot(1:15, SVMMeanRA(1,:), 'r:',1:15, SVMMeanRA(2,:), 'g:',1:15,...
%     SVMMeanRA(3,:), 'b:', 'LineWidth',1.5); hold on;
% plot(1:15, SVMMeanRB(1,:), 'r--',1:15, SVMMeanRB(2,:), 'g--',1:15,...
%     SVMMeanRB(3,:), 'b--', 'LineWidth',1.5) 
xlabel('Number of Synergies','interpreter','latex','FontSize',18,'FontWeight','bold')
ylabel('Performance','interpreter','latex','FontSize',18,'FontWeight','bold')
legend('FREE','Grasp', 'ASL','FREE - RA','Grasp - RA', 'ASL - RA', 'FREE - RB','Grasp - RB', 'ASL - RB')
title('ASL Classification','interpreter','latex','FontSize',18,'FontWeight','bold')
grid on;

subplot(1,2,2);
errorbar(1:15, SVMMean(4,:), SVMSTD(4,:),'r-*'); hold on;
errorbar(1:15, SVMMean(5,:), SVMSTD(5,:),'-*','color',[0, 0.5, 0]); hold on;
errorbar(1:15, SVMMean(6,:), SVMSTD(6,:),'b-*'); hold on;
errorbar(1:15, SVMMeanRA(4,:), SVMSTDRA(4,:),'r:'); hold on;
errorbar(1:15, SVMMeanRA(5,:), SVMSTDRA(5,:),':','color',[0, 0.5, 0]); hold on;
errorbar(1:15, SVMMeanRA(6,:), SVMSTDRA(6,:),'b:'); hold on;
errorbar(1:15, SVMMeanRB(4,:), SVMSTDRB(4,:),'r--'); hold on;
errorbar(1:15, SVMMeanRB(5,:), SVMSTDRB(5,:),'--','color',[0, 0.5, 0]); hold on;
errorbar(1:15, SVMMeanRB(6,:), SVMSTDRB(6,:),'b--'); hold on;

xlim([0 16])
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]);
% t = plot(1:16, SVMMeanRA(4,:), 'r:',1:16, SVMMeanRA(5,:), 'g:',1:16,...
%     SVMMeanRA(6,:), 'b:', 'LineWidth',1.5); hold on;
% k = plot(1:16, SVMMeanRB(4,:), 'r--',1:16, SVMMeanRB(5,:), 'g--',1:16,...
%     SVMMeanRB(6,:), 'b--', 'LineWidth',1.5) ;
xlabel('Number of Synergies','interpreter','latex','FontSize',18,'FontWeight','bold')
ylabel('Performance','interpreter','latex','FontSize',18,'FontWeight','bold')
%legend([h(1) t(1) k(1)],{'Actual Results','Random Activation', 'Random Base'})
title('Grasp Mimicking Classification','interpreter','latex','FontSize',18,'FontWeight','bold')
grid on;
ylim([0 1])


% 1-perfASLtoGraspRandom
% 2-perfPronedtoGraspRandom,
% 3-perfSemiPronedtoGraspRandom,
% 4-perfNaturaltoGraspRandom,
% 5-perfGrasptoGraspRandom,
% 6-perfASLtoASLRandom,
% 7-perfPronedtoASLRandom,
% 8-perfSemiPronedtoASLRandom,
% 9-perfNaturaltoASLRandom,
% 10-perfGrasptoASLRandom,
% 11-perfASLtoPronedRandom,
% 12-perfPronedtoPronedRandom,
% 13-perfSemiPronedtoPronedRandom,
% 14-perfNaturaltoPronedRandom,
% 15-perfGrasptoPronedRandom,
% 20-perfASLtoSemiPronedRandom,
% 21-perfPronedtoSemiPronedRandom,
% 22-perfSemiPronedtoSemiPronedRandom,
% 23-perfNaturaltoSemiPronedRandom,
% 24-perfGrasptoSemiPronedRandom