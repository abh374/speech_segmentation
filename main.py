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

plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

fn = sys.argv[1]		#audio file name
sr = int(sys.argv[2])		#sampling rate of the audio

loader=essentia.standard.MonoLoader(filename =fn ,sampleRate = sr)
audio = loader()		#audio sequence

#resample the audio to 16kHz
rs = Resample(inputSampleRate = sr, outputSampleRate = 20000 , quality = 0)
audio = rs(audio)
sr = 20000

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
wlf_silence = False		#stores whether or not last frame was silent

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
		if(len(spectogram_temp) > 1):
			features.append(spectogram_temp)
		spectogram_temp = []

	fstart = fstart + overlap
	fstop = fstop + overlap

for i in range (0, len(features)):
	img = gen_spectogram_image(features[i])
	cv2.imshow("asdf",img)

spectogram = np.array(spectogram)
img = gen_spectogram_image(spectogram)
img1 = gen_spectogram_image(spectogram1)
cv2.imshow("all",img)
cv2.imshow("speech",img1)
cv2.waitKey(0)


cv2.destroyAllWindows()
