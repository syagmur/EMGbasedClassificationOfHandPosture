import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
plt.close('all')
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
colorNames = [name for hsv, name in by_hsv]


results = np.load('NER_SVM.npy')
mSubject = np.mean(results,0)
vSubject = np.std(results,0)
x = np.arange(1, 16, 1)


plt.figure(1)
plt.subplot(121)
plt.errorbar(x, mSubject[3,:],  yerr=vSubject[3,:], linestyle='--', marker='s', markersize=7, linewidth=2.8, color=colorNames[123])
plt.errorbar(x, mSubject[5,:],  yerr=vSubject[5,:], linestyle='--', marker='p', markersize=7, linewidth=2.8, color=colorNames[153])
plt.errorbar(x, mSubject[1,:],  yerr=vSubject[1,:], linestyle='--', marker='v', markersize=7, linewidth=2.8, color=colorNames[65])
plt.errorbar(x, mSubject[9,:],  yerr=vSubject[9,:], linestyle=':', marker='s', markersize=7, linewidth=2.5, color=colorNames[123])
plt.errorbar(x, mSubject[11,:],  yerr=vSubject[11,:], linestyle=':', marker='p', markersize=7, linewidth=2.5, color=colorNames[153])
plt.errorbar(x, mSubject[7,:],  yerr=vSubject[7,:], linestyle=':', marker='v', markersize=7, linewidth=2.5, color=colorNames[65])
plt.errorbar(x, mSubject[15,:],  yerr=vSubject[15,:], linestyle='-.', marker='s', markersize=7, linewidth=2.2, color=colorNames[123])
plt.errorbar(x, mSubject[17,:],  yerr=vSubject[17,:], linestyle='-.', marker='p', markersize=7, linewidth=2.2, color=colorNames[153])
plt.errorbar(x, mSubject[13,:],  yerr=vSubject[13,:], linestyle='-.', marker='v', markersize=7, linewidth=2.2, color=colorNames[65])
# plt.legend(['Free', 'Grasp', 'ASL', 'Free - RB', 'Grasp - RB', 'ASL - RB', 'Free - RA', 'Grasp - RA', 'ASL - RA'], loc='lower center')
plt.title('ASL', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.xlabel('Number of Synergies', fontsize=14)
plt.text(0.19, 0.03, 'a) ASL Classification', fontsize=18, transform=plt.gcf().transFigure)
plt.xticks(np.arange(0, 16, step=1))
plt.yticks(np.arange(0, 0.95, step=.1))
plt.grid(True)
plt.subplot(122)
plt.errorbar(x, mSubject[2,:],  yerr=vSubject[2,:], linestyle='--', marker='s', markersize=7, linewidth=2.8, color=colorNames[123])
plt.errorbar(x, mSubject[0,:],  yerr=vSubject[0,:], linestyle='--', marker='p', markersize=7, linewidth=2.8, color=colorNames[153])
plt.errorbar(x, mSubject[4,:],  yerr=vSubject[4,:], linestyle='--', marker='v', markersize=7, linewidth=2.8, color=colorNames[65])
plt.errorbar(x, mSubject[8,:],  yerr=vSubject[8,:], linestyle=':', marker='s', markersize=7, linewidth=2.5, color=colorNames[123])
plt.errorbar(x, mSubject[6,:],  yerr=vSubject[6,:], linestyle=':', marker='p', markersize=7, linewidth=2.5, color=colorNames[153])
plt.errorbar(x, mSubject[10,:],  yerr=vSubject[10,:], linestyle=':', marker='v', markersize=7, linewidth=2.5, color=colorNames[65])
plt.errorbar(x, mSubject[14,:],  yerr=vSubject[14,:], linestyle='-.', marker='s', markersize=7, linewidth=2.2, color=colorNames[123])
plt.errorbar(x, mSubject[12,:],  yerr=vSubject[12,:], linestyle='-.', marker='p', markersize=7, linewidth=2.2, color=colorNames[153])
plt.errorbar(x, mSubject[16,:],  yerr=vSubject[16,:], linestyle='-.', marker='v', markersize=7, linewidth=2.2, color=colorNames[65])
plt.title('Grasp', fontsize=18)
plt.text(0.70, 0.03, 'b) Grasp Classification', fontsize=18, transform=plt.gcf().transFigure)
plt.ylabel('Accuracy', fontsize=18)
plt.xlabel('Number of Synergies', fontsize=14)
plt.xticks(np.arange(0, 16, step=1))
plt.yticks(np.arange(0, 0.95, step=.1))
plt.figlegend(['Free', 'Grasp', 'ASL', 'Free - RB', 'Grasp - RB', 'ASL - RB', 'Free - RA', 'Grasp - RA', 'ASL - RA'], loc =9, bbox_to_anchor=(0.5,0.89))
plt.subplots_adjust(left=0.04, right=0.98)
plt.grid(True)
plt.show()