import essentia
from essentia.standard import *
import cv2
import sys
import os
import numpy as np
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
from custom_methods import *
import math
from skimage.transform import radon, iradon

#plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

fn = sys.argv[1]		#audio file name
sr = int(sys.argv[2])		#sampling rate of the audio

loader=essentia.standard.MonoLoader(filename =fn, sampleRate = sr )
audio = loader()		#audio sequence

#resample the audio to 16kHz
if(sr>20000):

	rs = Resample(inputSampleRate = sr, outputSampleRate = 16000 , quality = 0)
	audio = rs(audio)
	sr = 16000

n_samples = len(audio)		#number of samples in the audio


w =  Windowing(type = 'hann')
wl = sr/1000*25			#window length considering a window of 25ms
overlap = sr/1000*15		#overlap between two consecutive windows 10ms


spectrum = Spectrum(size=wl)
energy = Energy()
ener=[]

spectogram = []
fstart = 0
fstop = fstart+wl
while fstop < n_samples:
	frame = audio[fstart:fstop]
	e = math.log10(energy(frame))
	ener.append(e)		#log of total energy is being taken

	fstart = fstart + overlap
	fstop = fstop + overlap


#gen_energy_histogram(ener)		#draw and plot the histogram of energies

temp = np.array(ener)
temp = temp.reshape(-1,1)		#clustering demands the matrix in certain fashion
threshold_gmm , threshold_cluster = estimate_threshold_VAD(temp)	#threshold contains various GMM parametes

#use predict_proba(x) to evaluate posteriori

#we will use the GMM to evaluate the VAD
#this variable sores mean[0] < mean[1] of GMM 
m = threshold_gmm.means_[0] < threshold_gmm.means_[1]

spectogram = []			#this contains spectogram of all the frames
spectogram1 = []		#this contaisn spectogram of only speech portions
features = []			#this contains the spectrum of voiced portion of speech in 
				#in sliced fashion
spectogram_temp = []
wlf_silent = True		#stores whether or not last frame was silent

trans_sample = []		#stores the transitioning samples speech->silence and vice-versa

fstart = 0
fstop = fstart+wl
while fstop < n_samples:
	frame = audio[fstart:fstop]
	ener = [math.log10(energy(frame))]		#log of total energy is being taken
	spec = spectrum(w(frame))

	a = threshold_cluster.predict([ener])
	b = threshold_gmm.predict_proba([ener])
	
	spectogram.append(spec)

	#write your code here to separate the two spectograms
	if( (b[0][0] > b[0][1] and not(m)) or  (b[0][0] < b[0][1] and m)):
		spectogram1.append(spec)

		if(not(wlf_silent)):
			spectogram_temp.append(spec)
		wlf_silent = False

	else :
		wlf_silent = True
		if(len(spectogram_temp) > 0):

			trans_sample.append(fstart-overlap)

			features.append(spectogram_temp)
		spectogram_temp = []


	fstart = fstart + overlap
	fstop = fstop + overlap



#merging the frames
i = 0
while ( i < len(features)-1):

	diff = trans_sample[i+1]-trans_sample[i]
	len_frame_ip1 = len(features[i+1])
	n_sample_ip1 = 15*len_frame_ip1 + 10
	if(diff - n_sample_ip1 < 16000):
		features[i] = features[i] + features[i+1]
		features.pop(i+1)
		trans_sample.pop(i)
	else:
		i = i+1


print trans_sample
first = True

corarray = [[],[],[],[],[],[],[]]
for i in range (0, len(features)):
	img = np.array(features[i])

	projections1 = radon(img, theta=[22.5]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections2 = radon(img, theta=[45]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections3 = radon(img, theta=[67.5]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections4 = radon(img, theta=[90]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections5 = radon(img, theta=[112.5]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections6 = radon(img, theta=[135]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections7 = radon(img, theta=[157.5]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	
	print "Current sample window: ", trans_sample[i]
	#print projections.shape
	#plt.plot(projections)

	if(first is False):
		corr1 = np.corrcoef(a[0].T,projections1.T)
		corr2 = np.corrcoef(a[1].T,projections2.T)
		corr3 = np.corrcoef(a[2].T,projections3.T)
		corr4 = np.corrcoef(a[3].T,projections4.T)
		corr5 = np.corrcoef(a[4].T,projections5.T)
		corr6 = np.corrcoef(a[5].T,projections6.T)
		corr7 = np.corrcoef(a[6].T,projections7.T)

		print corr1
		corarray[0].append( (corr1[0][1]))
		corarray[1].append( (corr2[0][1]))
		corarray[2].append( (corr3[0][1]))
		corarray[3].append( (corr4[0][1]))
		corarray[4].append( (corr5[0][1]))
		corarray[5].append( (corr6[0][1]))
		corarray[6].append( (corr7[0][1]))
	a = np.stack((projections1,projections2,projections3,projections4,projections5,projections6,projections7))


	cv2.imshow("asdf",img)
	cv2.waitKey(0)
	#show()
	first = False

spectogram = np.array(spectogram)
img = gen_spectogram_image(spectogram)
img1 = gen_spectogram_image(spectogram1)

height, width = img.shape[:2]
img = np.rot90(img)


plt.subplot(10 ,1, 1)
plt.imshow(img)
plt.title('COmpare')
plt.ylabel('Spectrogram')
plt.xlabel('Pixel')

plt.subplot(10,1, 2)
plt.imshow(np.rot90(img1))
plt.xlabel('time (s)')
plt.ylabel('Amplitude')

plt.subplot(10,1, 3)
plt.plot(audio[:])
plt.xlabel('time (s)')
plt.ylabel('Amplitude')

plt.subplot(10 , 1, 4)
plt.plot(corarray[0])
plt.xlabel('window')
plt.ylabel('correlation1')

plt.subplot(10 , 1, 5)
plt.plot(corarray[1])
plt.xlabel('window')
plt.ylabel('correlation2')


plt.subplot(10, 1, 6)
plt.plot(corarray[2])
plt.xlabel('window')
plt.ylabel('correlation3')


plt.subplot(10, 1, 7)
plt.plot(corarray[3])
plt.xlabel('window')
plt.ylabel('correlation4')


plt.subplot(10, 1, 8)
plt.plot(corarray[4])
plt.xlabel('window')
plt.ylabel('correlation5')


plt.subplot(10 , 1, 9)
plt.plot(corarray[5])
plt.xlabel('window')
plt.ylabel('correlation6')


plt.subplot(10, 1, 10)
plt.plot(corarray[6])
plt.xlabel('window')
plt.ylabel('correlation7')
plt.show()

#cv2.imshow("all",img)
#cv2.imshow("speech",img1)
cv2.waitKey(0)


cv2.destroyAllWindows()
