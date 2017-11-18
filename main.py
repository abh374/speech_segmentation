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

loader=essentia.standard.MonoLoader(filename =fn, sampleRate = sr )
audio = loader()		#audio sequence
n_samples = len(audio)		#number of samples in the audio


w =  Windowing(type = 'hann')
wl = sr/1000*25			#window length considering a window of 25ms
overlap = sr/1000*10		#overlap between two consecutive windows 10ms


spectrum = Spectrum(size = wl)
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
threshold = estimate_threshold_VAD(temp)	#threshold contains various GMM parametes
print threshold.labels_
						#use predict_proba(x) to evaluate posteriori

spectogram = []			#this contains spectogram of all the frames
spectogram1 = []		#this contaisn spectogram of only speech portions
fstart = 0
fstop = fstart+wl
while fstop < n_samples:
	frame = audio[fstart:fstop]
	ener = [math.log10(energy(frame))]		#log of total energy is being taken
	spec = spectrum(w(frame))

	a = threshold.predict([ener])
	print a
	spectogram.append(spec)
	#write your code here to separate the two spectograms
	#the one written below is hardcoded one
	if(a[0] == 0):
		spectogram1.append(spec)

	fstart = fstart + overlap
	fstop = fstop + overlap


spectogram = np.array(spectogram)
img = gen_spectogram_image(spectogram)
img1 = gen_spectogram_image(spectogram1)

height, width = img.shape[:2]
img = np.rot90(img)


plt.subplot(2 ,1, 1)
plt.imshow(img)
plt.title('COmpare')
plt.ylabel('Spectrogram')
plt.xlabel('Pixel')



plt.subplot(2,1, 2)
plt.plot(audio[:])
plt.xlabel('time (s)')
plt.ylabel('Amplitude')

plt.show()

#cv2.imshow("all",img)
#cv2.imshow("speech",img1)
cv2.waitKey(0)


cv2.destroyAllWindows()
