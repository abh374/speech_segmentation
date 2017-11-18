import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import mixture

def gen_spectogram_image(spec):
	rows = len(spec)
	cols = len(spec[0])
	img = np.zeros((rows,cols),dtype= np.uint8)

	'''
	assuming the values in the trasform varies from 10^-14 to 10^2
	the corresponding non-linear mapping gave the equation
	y = 16 log10(x) + 224
	'''
	for i in range(0,rows):
		for j in range(0,cols):
			try:
				img[i][j] = int(math.log10(spec[i][j]) * 16 + 224)
			except ValueError:
				img[i][j] = 0

	return img

def gen_energy_histogram(ener):
	'''normally log of energy for a frame is not more than 3 and for many
	silence parts it is as low as -8 so bins from bins have been taken 
	[-12 , -12 + 0.17d], here d = {0,1,2,...,99}'''

	y = np.linspace(-12 , 5 , num=100)
	plt.hist(ener , y , alpha = 0.5)
	plt.show()

def estimate_threshold_VAD(ener):
	'''it was observed that the log energies of the frame approximates somewhat
	a mixture of two gaussians or two clusters, and the same was used to estimate 
	the threshold
	'''
	N = 2

	CV = 'full'
	gmm = mixture.GaussianMixture(n_components = N , covariance_type = CV)
	gmm.fit(ener)


	kmeans = KMeans(n_clusters = N , random_state = 0).fit(ener)
	return gmm , kmeans
