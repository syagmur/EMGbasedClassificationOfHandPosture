clear all; 

path2 = strcat('C:\Users\gunay\Desktop\HANDS\Data\CM')

baseASL = matrixToBeSaved(1,:);
baseGrasp = matrixToBeSaved(2,:);
baseProned = matrixToBeSaved(3,:);
baseSemiProned = matrixToBeSaved(4,:);
baseNatural = matrixToBeSaved(5,:); 
activationASL = matrixToBeSaved(6,:);
activationGrasp = matrixToBeSaved(7,:);
activationProned = matrixToBeSaved(8,:);
activationSemiProned = matrixToBeSaved(9,:);
activationNatural = matrixToBeSaved(10,:);
randASLBase = matrixToBeSaved(11,:);
randGraspBase = matrixToBeSaved(12,:);
randPronedBase = matrixToBeSaved(13,:);
randSemiPronedBase = matrixToBeSaved(14,:);
randNaturalBase = matrixToBeSaved(15,:);
randASLActivation = matrixToBeSaved(16,:);
randGraspActivation = matrixToBeSaved(17,:);
randPronedActivation = matrixToBeSaved(18,:);
randSemiPronedActivation = matrixToBeSaved(19,:);
randNaturalActivation = matrixToBeSaved(20,:);
randASLBaseShuffled = matrixToBeSaved(21,:); 
randGraspBaseShuffled = matrixToBeSaved(22,:);
randPronedBaseShuffled = matrixToBeSaved(23,:);
randSemiPronedBaseShuffled = matrixToBeSaved(24,:);
randNaturalBaseShuffled = matrixToBeSaved(25,:);

addpath(genpath(path2));  

NaturalMovement1 = load(strcat(path2,'\FreeMovement\FreeMovement1.mat'));
NaturalMovement2 = load(strcat(path2,'\FreeMovement\FreeMovement2.mat'));
NaturalMovement3 = load(strcat(path2,'\FreeMovement\FreeMovement3.mat'));
ASLgestures1 = dir(fullfile(strcat(path2,'\ASL\Trial1'),'*.mat'));
ASLgestures2= dir(fullfile(strcat(path2,'\ASL\Trial2'),'*.mat'));
ASLgestures3 = dir(fullfile(strcat(path2,'\ASL\Trial3'),'*.mat'));
Graspgestures1 = dir(fullfile(strcat(path2,'\GRASP\Trial1'),'*.mat'));
Graspgestures2= dir(fullfile(strcat(path2,'\GRASP\Trial2'),'*.mat'));
Graspgestures3 = dir(fullfile(strcat(path2,'\GRASP\Trial3'),'*.mat'));
MVCs= dir(fullfile(strcat(path2,'\MVCs'),'*.mat'));
nChannels = size(NaturalMovement1.EMG,2);
nCross = 10;
initial = 3500;
final = 2300;

for i=1:size(ASLgestures1)
    matName = strcat(path2,'\ASL\Trial1\',ASLgestures1(i,:).name);
    matName2 = strcat(path2,'\ASL\Trial2\',ASLgestures2(i,:).name);
    matName3 = strcat(path2,'\ASL\Trial3\',ASLgestures3(i,:).name);
    Trial{1,i} = load(matName); 
    ASLData{1,i} = Trial{1,i}.EMG;
    Trial{2,i} = load(matName2); 
    ASLData{2,i} = Trial{2,i}.EMG;
    Trial{3,i} = load(matName3); 
    ASLData{3,i} = Trial{3,i}.EMG;
end

for i=1:size(Graspgestures1)
    matName = strcat(path2,'\GRASP\Trial1\',Graspgestures1(i,:).name);
    matName2 = strcat(path2,'\GRASP\Trial2\',Graspgestures2(i,:).name);
    matName3 = strcat(path2,'\GRASP\Trial3\',Graspgestures3(i,:).name);
    Trial{1,i} = load(matName); 
    GraspData{1,i} = Trial{1,i}.EMG;
    Trial{2,i} = load(matName2); 
    GraspData{2,i} = Trial{2,i}.EMG;
    Trial{3,i} = load(matName3); 
    GraspData{3,i} = Trial{3,i}.EMG;
    if mod(i,2) == 1
       PronedData{1, ceil(i/2)} = GraspData{1,i};
       PronedData{2, ceil(i/2)} = GraspData{2,i};
       PronedData{3, ceil(i/2)} = GraspData{3,i};
    else 
       SemiPronedData{1, i/2} = GraspData{1,i};
       SemiPronedData{2, i/2} = GraspData{2,i};
       SemiPronedData{3, i/2} = GraspData{3,i};
    end
end

% % Maximum voluntary contraction data
% MVCData{1} = [];
% for i=1:size(MVCs)
%     matName = strcat(path2,'\MVCs\', MVCs(i,:).name)
%     Trial = load(matName); 
%     MVCData{1} = [MVCData{1} Trial.AllData(:,i)];
% end

% Natural movement data: arbitrarily exploring the space 
NaturalData = [NaturalMovement1.EMG;NaturalMovement2.EMG;NaturalMovement3.EMG];

% Calculate RMS values
% [ampMVC, labelMVS] = rmsCalculator(MVCData, 1, 300);
[rmsProned, labelProned] = rmsCalculator(PronedData, initial, final);
[rmsSemiProned, labelSemiProned] = rmsCalculator(SemiPronedData, initial, final);
[rmsASL, labelASL] = rmsCalculator(ASLData, initial, final);
[rmsGrasp, labelGrasp] = rmsCalculator(GraspData, initial, final);

% % Normalize wrt the MVC amplitude
% ampProned = MVCnorm(rmsProned, ampMVC);
% ampSemiProned = MVCnorm(rmsSemiProned, ampMVC);
% ampASL = MVCnorma(rmsASL, ampMVC);
% ampGrasp = MVCnorm(rmsGrasp, ampMVC);
ampProned = rmsProned;
ampSemiProned =rmsSemiProned;
ampASL = rmsASL;
ampGrasp = rmsGrasp;


classificationAll();
classificationRandomActivation();
classificationRandomBase();
classificationRandomBaseShuffled();
