import sys
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
from os import walk
from scipy import signal
from scipy.signal import butter, lfilter
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn_extensions.extreme_learning_machines import ELMClassifier
from sklearn.model_selection import KFold
from scipy import optimize
from scipy.integrate import quad
from sklearn import svm
import time
import itertools
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import csv
import pickle
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
plt.close('all')
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
colorNames = [name for hsv, name in by_hsv]
sys.path.append('/Users/gunay/Desktop/HANDS/Codes/hands')


# butterworth bandpass filter initialization
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# butterworth bandpass filter implementation
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Apply notch filter at 60 Hz frequency
def notchFilter(data):
    F0 = 60.0
    normFreq = F0/(Fs/2)
    b, a = signal.iirnotch(normFreq, 30)
    y = signal.filtfilt(b, a, butter_bandpass_filter(data, 0.1, 400, Fs, 5))
    return y


# calculate the rms values of the data for the data given between initial and final values
def RMSCal(data, initial, final, maxRMS = []):
    sizeCheck = data.shape      # check if the data is free movement or MVC or others
    rmsArray = []
    labelArray = []
    filteredData = notchFilter(data)
    if np.size(sizeCheck) == 3:        # Free movement
        for i in range(0, trialSize):
            trialLength = 0
            while trialLength < sizeCheck[1]:
                temp = np.sqrt(np.mean(np.square(filteredData[i, trialLength:(trialLength+nSampleSize),:]), axis=0))
                rmsArray.append(temp)
                trialLength = trialLength + buffer + nSampleSize
        return rmsArray/maxRMS
    elif np.size(sizeCheck) == 2:     # MVC
        trialLength = 0
        while trialLength < sizeCheck[0]:
            temp = np.sqrt(np.mean(np.square(filteredData[trialLength:(trialLength + nSampleSize), :]), axis=0))
            rmsArray.append(temp)
            trialLength = trialLength + buffer + nSampleSize
        return rmsArray
    elif np.size(sizeCheck) == 4:     # ASL or Grasp or SemiProned or Proned
        for gesture in range(0,sizeCheck[0]):
            for trial in range(0, trialSize):
                trialLength = initial
                while trialLength < sizeCheck[2]-final:
                    temp = np.sqrt(np.mean(np.square(filteredData[gesture, trial, trialLength:(trialLength + 300), :]), axis=0))
                    rmsArray.append(temp)
                    trialLength = trialLength + buffer + nSampleSize
                    labelArray.append(gesture)
        return rmsArray/maxRMS, np.array(labelArray)

# Apply NMF for the given rank and return base and activation matrices
def synergyBaseSearcher(data, rnk):
    model = NMF(n_components=rnk, init='random', random_state=0)
    baseMatrix = model.fit_transform(np.transpose(data))
    activationMatrix = model.components_
    return baseMatrix, activationMatrix

# Within classification implementation
# data: rms values of a data
# label: labels of the data
def activationClassifier(data, label):
    crossVal = KFold(n_splits=nCross, shuffle=True)
    classifier = ELMClassifier(n_hidden=150, activation_func='sigmoid')
    acc = np.zeros([nCross, nChannel - 1])
    # Run the classifier for each possible rank selection
    for rnk in range(0, nChannel - 1):
        cnt = 0
        for train_index, test_index in crossVal.split(data[rnk]):
            trainData, testData = data[rnk][train_index, :], data[rnk][test_index, :]
            trainLabel, testLabel = label[train_index], label[test_index]
            classifier.fit(trainData, trainLabel)
            predictedLabel = classifier.predict(testData)
            cf = confusion_matrix(testLabel, predictedLabel)
            acc[cnt, rnk] = accuracy_score(testLabel, predictedLabel)
            cnt = cnt + 1
    return acc, cf

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Cross classification
# base: the base matrix to map the data for finding activation values
# data: rms values of one of the datasets
# label: labels of the data
def crossClassifier(base, data, label):
    activation = []
    data = np.array(data)
    for rnk in range(0, nChannel - 1):
        temp = np.zeros([np.shape(data)[0], rnk+1])
        for i in range(0, np.shape(data)[0]):
            temp[i, :] = optimize.nnls(base[rnk], np.reshape(data[i,:],(nChannel,)))[0]
        activation.append(temp)
    acc, cf = activationClassifier(activation, label)
    return acc


def synDec(crossAcc):
    # Curve fitting for the accuracy values
    acc = np.mean(crossAcc, 0)
    # x1 = np.linspace(0, 1, 15)
    x1 = np.arange(0, 15, 1)
    cur = np.poly1d(np.polyfit(x1, acc, 3))
    f1 = cur(x1)
    x1_new = np.linspace(0,14,15)
    # plt.plot(x1_new, f1, linestyle='-.', linewidth=3.0)
    # plt.plot(acc)

    area = np.zeros([15,2])
    for init in range(0,15):
        x2 = np.linspace(init, 16, 2)
        acc2 = np.array([acc[init], acc[14]])
        lin = np.poly1d(np.polyfit(x2, acc2, 1))
        x2_new = np.linspace(init, 14, 15)
        f2 = lin(x2_new)
        # func1 = lambda x: cur[3] * x ** 3 + cur[2] * x ** 2 + cur[1] * x + cur[0]
        # func2 = lambda x: lin[1] * x + lin[0]
        func1 = lambda x: cur(x)
        func2 = lambda x: lin(x)
        h = lambda x: np.abs(func1(x) - func2(x))
        area[init, :] = quad(h, init, 14)
        # plt.plot(x2_new, f2)
        # plt.show()

    val = -1;
    for i in range(0,14):
        if area[i,0] - area[i+1,0]<= 1e-2:
            val = i
            break

    # plt.legend(['Fitted Curve','Real Line','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'],loc='lower right')
    # print(area, val)
    return val


def randomGenerator(base):
    randomBase = []
    for syn in range(0,15):
        mVal = np.mean(base[syn])
        vVal = np.var(base[syn])
        [r,c] = np.shape(base[syn])
        randomBase.append(np.random.normal(mVal, vVal,[r,c]))
    return randomBase


def crossClassifierRandomActivation(base, data, label):
    activation = []
    data = np.array(data)
    for rnk in range(0, nChannel - 1):
        temp = np.zeros([np.shape(data)[0], rnk+1])
        for i in range(0, np.shape(data)[0]):
            temp[i, :] = optimize.nnls(base[rnk], np.reshape(data[i,:],(nChannel,)))[0]
        activation.append(temp)
    acc, cf = activationClassifier(randomGenerator(activation), label)
    return acc

start_time = time.time()
# Global variables
subPath = 'C:/Users/gunay/Desktop/HANDS/Data/AP/'       # path for the subject
nChannel = 16       # Number of EMG channels
trialLength = 10000         # Length of each trial (10s * 1000 sampling rate)
trialSize = 3       # number of trials for each data collection
initial = 3500      # starting point for rms calculation (first 2 seconds is rest and 1.5 sec for gesture initialization)
final = 2300        # stopping point for rms calculation (last 2 seconds for relaxing)
nCross = 10         # k-fold cross-validation
nSampleSize = 300  # rms block width
buffer = 100  # rms buffer width
Fs = 1000.0     # sampling frequency
subList = ['AP', 'CM', 'DS', 'GP', 'MV']
allELMResults = np.zeros((5, 18, 15))

for cur in range(0,5):
    subPath = 'C:/Users/gunay/Desktop/HANDS/Data/' + subList[cur]      # path for the subject

    # Get ASL file names from the subject's data path
    ASLfileNames = []
    for (dirpath, dirnames, filenames) in walk(subPath + '/ASL/Trial1'):
        ASLfileNames.extend(filenames)

    # Get Grasp file names from the subject's data path
    GraspfileNames = []
    for (dirpath, dirnames, filenames) in walk(subPath + '/Grasp/Trial1'):
        GraspfileNames.extend(filenames)

    # Get MVC file names from the subject's data path
    MVCfileNames = []
    for (dirpath, dirnames, filenames) in walk(subPath + '/MVCs'):
        MVCfileNames.extend(filenames)

    ASLclassNo = np.shape(ASLfileNames)[0]
    GraspclassNo = np.shape(GraspfileNames)[0]

    # Get all ASL data from the accessed ASL files
    ASLData = np.zeros([ASLclassNo, trialSize, trialLength, nChannel])
    for i in range(0, ASLclassNo):
        for j in range(0, trialSize):
            mats = []
            dataDir = subPath + '/ASL/Trial' + str(j + 1) + '/' + ASLfileNames[i]
            mats = loadmat(dataDir)['EMG']
            ASLData[i, j, :, :] = mats

    # Get all Grasp data from the accessed Grasp files and separate the data as proned and semi-proned
    GraspData = np.zeros([GraspclassNo, trialSize, trialLength, nChannel])
    PronedGraspData = np.zeros([GraspclassNo / 2, trialSize, trialLength, nChannel])
    SemiPronedGraspData = np.zeros([GraspclassNo / 2, trialSize, trialLength, nChannel])
    for i in range(0, GraspclassNo):
        for j in range(0, trialSize):
            mats = []
            dataDir = subPath + '/Grasp/Trial' + str(j + 1) + '/' + GraspfileNames[i]
            mats = loadmat(dataDir)['EMG']
            GraspData[i, j, :, :] = mats
            if i % 2 == 0:
                PronedGraspData[i / 2, j, :, :] = mats
            else:
                SemiPronedGraspData[i / 2, j, :, :] = mats

    # Get all free movement data from the subject's folder
    FreeData = np.zeros([trialSize, 120000, nChannel])
    for i in range(0, trialSize):
        mats = []
        dataDir = subPath + '/FreeMovement/Trial' + str(i + 1) + '/' + 'FreeMovement'
        mats = loadmat(dataDir)['EMG']
        FreeData[i, :, :] = mats

    # Get all maximum voluntary contraction data from the subject's folder
    MVCData = []
    for i in range(0, np.size(MVCfileNames)):
        mats = []
        dataDir = subPath + '/MVCs' + '/' + MVCfileNames[i]
        mats = loadmat(dataDir)['AllData']
        MVCData.extend(mats)

    rmsMVC = RMSCal(np.array(MVCData), 0, nSampleSize)

    # Calculate the max of EMG and RMS for each muscle
    maxMVC = np.max(MVCData, 0)
    maxRMS = np.max(rmsMVC, 0)

    # Calculate RMS values for each dataset
    rmsProned, labelProned = RMSCal(PronedGraspData, initial, final, maxRMS=maxRMS)
    # rmsSemiProned, labelSemiProned = RMSCal(SemiPronedGraspData, initial, final, maxRMS=maxRMS)
    # rmsGrasp, labelGrasp = RMSCal(GraspData, initial, final, maxRMS=maxRMS)
    rmsASL, labelASL = RMSCal(ASLData, initial, final, maxRMS=maxRMS)
    rmsFree = RMSCal(FreeData, 0, nSampleSize, maxRMS=maxRMS)

    # Calculate base and activation values for all possible rank selection
    baseASL = [];
    baseProned = [];
    # baseSemiProned = [];
    # baseGrasp = [];
    baseFree = []
    activationASL = [];
    activationProned = [];
    # activationSemiProned = [];
    # activationGrasp = [];
    activationFree = []
    for rnk in range(1, nChannel):
        baseASLtemp, activationASLtemp = synergyBaseSearcher(rmsASL, rnk)
        basePronedtemp, activationPronedtemp = synergyBaseSearcher(rmsProned, rnk)
        # baseSemiPronedtemp, activationSemiPronedtemp = synergyBaseSearcher(rmsSemiProned, rnk)
        # baseGrasptemp, activationGrasptemp = synergyBaseSearcher(rmsGrasp, rnk)
        baseFreetemp, activationFreetemp = synergyBaseSearcher(rmsFree, rnk)
        baseASL.append(baseASLtemp);
        activationASL.append(np.transpose(activationASLtemp))
        baseProned.append(basePronedtemp);
        activationProned.append(np.transpose(activationPronedtemp))
        # baseSemiProned.append(baseSemiPronedtemp);
        # activationSemiProned.append(np.transpose(activationSemiPronedtemp))
        # baseGrasp.append(baseGrasptemp);
        # activationGrasp.append(np.transpose(activationGrasptemp))
        baseFree.append(baseFreetemp);
        activationFree.append(np.transpose(activationFreetemp))

    # Classification for within and cross cases
    # accGrasp, cfGrasp = activationClassifier(activationGrasp, labelGrasp)
    accProned, cfProned = activationClassifier(activationProned, labelProned)
    # accSemiProned, cfSemiProned = activationClassifier(activationSemiProned, labelSemiProned)
    accASL, cfASL = activationClassifier(activationASL, labelASL)
    print('Within-Class Classifications Completed')

    # accFreetoGrasp = crossClassifier(baseFree, rmsGrasp, labelGrasp)
    accFreetoProned = crossClassifier(baseFree, rmsProned, labelProned)
    # accFreetoSemiProned = crossClassifier(baseFree, rmsSemiProned, labelSemiProned)
    accFreetoASL = crossClassifier(baseFree, rmsASL, labelASL)
    print('Free-Based Classifications Completed')

    # accPronedtoGrasp = crossClassifier(baseProned, rmsGrasp, labelGrasp)
    # accPronedtoSemiProned = crossClassifier(baseProned, rmsSemiProned, labelSemiProned)
    accPronedtoASL = crossClassifier(baseProned, rmsASL, labelASL)
    print('Prone-Based Classifications Completed')
    #
    # accSemiPronedtoGrasp = crossClassifier(baseSemiProned, rmsGrasp, labelGrasp)
    # accSemiPronedtoProned = crossClassifier(baseSemiProned, rmsProned, labelProned)
    # accSemiPronedtoASL = crossClassifier(baseSemiProned, rmsASL, labelASL)
    # print('SemiProne-Based Classifications Completed')

    # accASLtoGrasp = crossClassifier(baseASL, rmsGrasp, labelGrasp)
    accASLtoProned = crossClassifier(baseASL, rmsProned, labelProned)
    # accASLtoSemiProned = crossClassifier(baseASL, rmsSemiProned, labelSemiProned)
    # print('ASL-Based Classifications Completed')

    # accGrasptoProned = crossClassifier(baseGrasp, rmsProned, labelProned)
    # accGrasptoASL = crossClassifier(baseGrasp, rmsASL, labelASL)
    # accGrasptoSemiProned = crossClassifier(baseGrasp, rmsSemiProned, labelSemiProned)
    print('Grasp Based Classifications Completed')

    randomBaseFree = randomGenerator(baseFree);
    randomBaseProned = randomGenerator(baseProned)
    # randomBaseSemiProned = randomGenerator(baseSemiProned)
    randomBaseASL = randomGenerator(baseASL)
    # randomBaseGrasp = randomGenerator(baseGrasp)

    # Classification for within and cross cases for random base condition
    # accGraspRandomBase = crossClassifier(randomBaseGrasp, rmsGrasp, labelGrasp)
    accPronedRandomBase = crossClassifier(randomBaseProned, rmsProned, labelProned)
    # accSemiPronedRandomBase = crossClassifier(randomBaseSemiProned, rmsSemiProned, labelSemiProned)
    accASLRandomBase = crossClassifier(randomBaseASL, rmsASL, labelASL)
    print('Random Base Within-Class Classifications Completed')

    # accFreetoGraspRandomBase = crossClassifier(randomBaseFree, rmsGrasp, labelGrasp)
    accFreetoPronedRandomBase = crossClassifier(randomBaseFree, rmsProned, labelProned)
    # accFreetoSemiPronedRandomBase = crossClassifier(randomBaseFree, rmsSemiProned, labelSemiProned)
    accFreetoASLRandomBase = crossClassifier(randomBaseFree, rmsASL, labelASL)
    print('Random Base Free-Based Classifications Completed')

    # accPronedtoGraspRandomBase = crossClassifier(randomBaseProned, rmsGrasp, labelGrasp)
    # accPronedtoSemiPronedRandomBase = crossClassifier(randomBaseProned, rmsSemiProned, labelSemiProned)
    accPronedtoASLRandomBase = crossClassifier(randomBaseProned, rmsASL, labelASL)
    # print('Random Base Prone-Based Classifications Completed')

    # accSemiPronedtoGraspRandomBase = crossClassifier(randomBaseSemiProned, rmsGrasp, labelGrasp)
    # accSemiPronedtoPronedRandomBase = crossClassifier(randomBaseSemiProned, rmsProned, labelProned)
    # accSemiPronedtoASLRandomBase = crossClassifier(randomBaseSemiProned, rmsASL, labelASL)
    # print('Random Base SemiProne-Based Classifications Completed')

    # accASLtoGraspRandomBase = crossClassifier(randomBaseASL, rmsGrasp, labelGrasp)
    accASLtoPronedRandomBase = crossClassifier(randomBaseASL, rmsProned, labelProned)
    # accASLtoSemiPronedRandomBase = crossClassifier(randomBaseASL, rmsSemiProned, labelSemiProned)
    print('Random Base ASL-Based Classifications Completed')

    # accGrasptoPronedRandomBase = crossClassifier(randomBaseGrasp, rmsProned, labelProned)
    # accGrasptoASLRandomBase = crossClassifier(randomBaseGrasp, rmsASL, labelASL)
    # accGrasptoSemiPronedRandomBase = crossClassifier(randomBaseGrasp, rmsSemiProned, labelSemiProned)
    print('Random Base Grasp Based Classifications Completed')

    # Classification for within and cross cases for random activation condition
    # accGraspRandomActivation = crossClassifierRandomActivation(baseGrasp, rmsGrasp, labelGrasp)
    accPronedRandomActivation = crossClassifierRandomActivation(baseProned, rmsProned, labelProned)
    # accSemiPronedRandomActivation = crossClassifierRandomActivation(baseSemiProned, rmsSemiProned, labelSemiProned)
    accASLRandomActivation = crossClassifierRandomActivation(baseASL, rmsASL, labelASL)
    print('Random Activation Within-Class Classifications Completed')

    # accFreetoGraspRandomActivation = crossClassifierRandomActivation(baseFree, rmsGrasp, labelGrasp)
    accFreetoPronedRandomActivation = crossClassifierRandomActivation(baseFree, rmsProned, labelProned)
    # accFreetoSemiPronedRandomActivation = crossClassifierRandomActivation(baseFree, rmsSemiProned, labelSemiProned)
    accFreetoASLRandomActivation = crossClassifierRandomActivation(baseFree, rmsASL, labelASL)
    print('Random Activation Free-Based Classifications Completed')

    # accPronedtoGraspRandomActivation = crossClassifierRandomActivation(baseProned, rmsGrasp, labelGrasp)
    # accPronedtoSemiPronedRandomActivation = crossClassifierRandomActivation(baseProned, rmsSemiProned, labelSemiProned)
    accPronedtoASLRandomActivation = crossClassifierRandomActivation(baseProned, rmsASL, labelASL)
    # print('Random Activation Prone-Based Classifications Completed')

    # accSemiPronedtoGraspRandomActivation = crossClassifierRandomActivation(baseSemiProned, rmsGrasp, labelGrasp)
    # accSemiPronedtoPronedRandomActivation = crossClassifierRandomActivation(baseSemiProned, rmsProned, labelProned)
    # accSemiPronedtoASLRandomActivation = crossClassifierRandomActivation(baseSemiProned, rmsASL, labelASL)
    # print('Random Activation SemiProne-Based Classifications Completed')

    # accASLtoGraspRandomActivation = crossClassifierRandomActivation(baseASL, rmsGrasp, labelGrasp)
    accASLtoPronedRandomActivation = crossClassifierRandomActivation(baseASL, rmsProned, labelProned)
    # accASLtoSemiPronedRandomActivation = crossClassifierRandomActivation(baseASL, rmsSemiProned, labelSemiProned)
    print('Random Activation ASL-Based Classifications Completed')

    # accGrasptoPronedRandomActivation = crossClassifierRandomActivation(baseGrasp, rmsProned, labelProned)
    # accGrasptoASLRandomActivation = crossClassifierRandomActivation(baseGrasp, rmsASL, labelASL)
    # accGrasptoSemiPronedRandomActivation = crossClassifierRandomActivation(baseGrasp, rmsSemiProned, labelSemiProned)
    print('Random Activation Grasp Based Classifications Completed')


    allELMResults[cur,0:2,:] = np.vstack((np.mean(accProned, 0), np.mean(accASL, 0)))
    allELMResults[cur,2:4,:] = np.vstack((np.mean(accFreetoProned, 0), np.mean(accFreetoASL, 0)))
    allELMResults[cur,4:6,:] = np.vstack((np.mean(accASLtoProned, 0), np.mean(accPronedtoASL, 0)))
    allELMResults[cur,6:8,:] = np.vstack((np.mean(accPronedRandomBase, 0), np.mean(accASLRandomBase, 0)))
    allELMResults[cur,8:10,:] = np.vstack((np.mean(accFreetoPronedRandomBase, 0), np.mean(accFreetoASLRandomBase, 0)))
    allELMResults[cur,10:12,:] = np.vstack((np.mean(accASLtoPronedRandomBase, 0), np.mean(accPronedtoASLRandomBase, 0)))
    allELMResults[cur,12:14,:] = np.vstack((np.mean(accPronedRandomActivation, 0), np.mean(accASLRandomActivation, 0)))
    allELMResults[cur,14:16,:] = np.vstack((np.mean(accFreetoPronedRandomActivation, 0), np.mean(accFreetoASLRandomActivation, 0)))
    allELMResults[cur,16:18,:] = np.vstack((np.mean(accASLtoPronedRandomActivation, 0), np.mean(accPronedtoASLRandomActivation, 0)))


    # Plot the results
    plt.figure(cur+1)
    plt.subplot(121)
    plt.plot(np.mean(accFreetoProned, 0), linestyle='--', marker='p', color=colorNames[123])
    plt.plot(np.mean(accProned, 0), linestyle='--', marker='*', color=colorNames[153])
    plt.plot(np.mean(accASLtoProned, 0), linestyle='--', marker='o', color=colorNames[65])
    plt.plot(np.mean(accFreetoPronedRandomBase, 0), linestyle=':', marker='p', color=colorNames[123])
    plt.plot(np.mean(accPronedRandomBase, 0), linestyle=':', marker='*', color=colorNames[153])
    plt.plot(np.mean(accASLtoPronedRandomBase, 0), linestyle=':', marker='o', color=colorNames[65])
    plt.plot(np.mean(accFreetoPronedRandomActivation, 0), linestyle='-.', marker='p', color=colorNames[123])
    plt.plot(np.mean(accPronedRandomActivation, 0), linestyle='-.', marker='*', color=colorNames[153])
    plt.plot(np.mean(accASLtoPronedRandomActivation, 0), linestyle='-.', marker='o', color=colorNames[65])
    # plt.legend(['Free', 'Proned', 'ASL'], loc='lower right')
    plt.title('Grasp')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Synergies')
    plt.xlim(0,16)
    plt.xticks(np.arange(0, 16, step=1))
    plt.yticks(np.arange(0, 0.95, step=.1))
    plt.grid(True)
    plt.subplot(122)
    plt.plot(np.mean(accFreetoASL, 0), linestyle='--', marker='p', color=colorNames[123])
    plt.plot(np.mean(accPronedtoASL, 0), linestyle='--', marker='*', color=colorNames[153])
    plt.plot(np.mean(accASL, 0), linestyle='--', marker='o', color=colorNames[65])
    plt.plot(np.mean(accFreetoASLRandomBase, 0), linestyle=':', marker='p', color=colorNames[123])
    plt.plot(np.mean(accPronedtoASLRandomBase, 0), linestyle=':', marker='*', color=colorNames[153])
    plt.plot(np.mean(accASLRandomBase, 0), linestyle=':', marker='o', color=colorNames[65])
    plt.plot(np.mean(accFreetoASLRandomActivation, 0), linestyle='-.', marker='p', color=colorNames[123])
    plt.plot(np.mean(accPronedtoASLRandomActivation, 0), linestyle='-.', marker='*', color=colorNames[153])
    plt.plot(np.mean(accASLRandomActivation, 0), linestyle='-.', marker='o', color=colorNames[65])
    plt.legend(['Free', 'Grasp', 'ASL', 'Free - RB', 'Grasp - RB', 'ASL - RB', 'Free - RA', 'Grasp - RA', 'ASL - RA'], loc='upper center')
    plt.title('ASL')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Synergies')
    plt.xticks(np.arange(0, 16, step=1))
    plt.yticks(np.arange(0, 0.95, step=.1))
    plt.grid(True)
    pltTitle = 'Results for Subject ' + str(cur + 1)
    plt.suptitle(pltTitle)
    plt.get_current_fig_manager().window.showMaximized()

np.save('NER_ELM', allELMResults)
plt.show()
savemat('NER_ELM.mat', mdict={'ELM': allELMResults})
