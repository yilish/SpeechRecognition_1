# import numpy
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
#
def read_wav(path: str):
    r'''
    :param path: The path of wave file to be read.
    :return: signal - the intensity of wave, sample rate - the rate of sampling
    '''


    sample_rate, signal = scipy.io.wavfile.read('Jiading.wav')
    plt.plot(signal)
    plt.show()
    return signal, sample_rate

def pre_emphasis(signal: np.ndarray, alpha):
    r'''
    This function is designed to do the pre-emphasis
    :param signal:
    :param alpha:
    :return:
    '''

if __name__ == '__main__':
    sig, sample_rate = read_wav('Jiading.wav')

