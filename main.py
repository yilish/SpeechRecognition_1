import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import scipy.fftpack


def read_wav(path: str) -> [np.ndarray, int]:
    r"""
    Reading the wav file, using scipy lib func.
    :param path: The path of wave file to be read.
    :return: signal - the intensity of wave, sample rate - the rate of sampling
    """
    sample_rate, signal = scipy.io.wavfile.read('Jiading.wav')
    return signal, sample_rate


def pre_emphasis(signal: np.ndarray, alpha: float = 0.97):
    r"""
    Eliminate the bias between high freq and low freq.
    :param signal: raw signal produced in the previous step
    :param alpha: a parameter to determine how much the previous signal is subtracted
    :return: pre-emphasized signal
    """
    res = signal[1:] - alpha * signal[:-1]  # y(t) = x(t) - a * x(t - 1)
    return res


def framing(signal: np.ndarray, sample_rate: int, frame_time: float = 0.025, stride: float = 0.010):
    r"""
    Creating several frames, to embed the pre-emphasized signal into frames.
    :param signal: previously-proceeded signal, a series.
    :param sample_rate: sample rate
    :param frame_time: the length of each frame
    :param stride: the step between frames
    :return: framed signal, whose shape is (frame_time * sample_rate, ceiling(signal / int(frame_stride * sample_rate)))
    """
    frame_len = int(frame_time * sample_rate)
    stride_len = int(stride * sample_rate)
    first_pos = 0
    framed_result = []
    while first_pos + frame_len < len(signal):
        framed_result.append(signal[first_pos: first_pos + frame_len])
        first_pos += stride_len
    last_frame = np.append(signal[first_pos:], np.zeros((frame_len - (len(signal) - first_pos))))
    framed_result.append(last_frame)
    framed_result = np.array(framed_result)
    return framed_result


def window(ori_frame: np.ndarray, N: int, alpha: float = 0.46164) -> np.ndarray:
    r"""
    Transforming the framed ones to continuous ones. Here we are using Hamming-like approaches.
    :param alpha: Windowing coefficient, where Hanning window is 0.5, Hamming window is 0.46164. Default: hamming.
    :param N: The length of Windowing.
    :param ori_frame: Original frame, which has been embedded previously.
    :return: Windowed(using certain weight redistributed) frames, whose shape is identical to the original ones.
    """
    n = np.arange(0, N)
    hamming = (1 - alpha) - alpha * np.cos(2 * np.pi * n / (N - 1))
    ori_frame *= hamming
    return ori_frame


def mel_filter_bank(low_freq: int, high_freq: int, nfilter: int, nfft: int, sample_rate: int) -> np.ndarray:
    r"""
    To non-linearize freq so that it shows more alike to human ear.
    Then creating some filters to extract triangle signals. (A kind of naive feature extraction)
    :param low_freq: The floor of freq, which is canonically zero.
    :param high_freq: The ceiling of freq, which is canonically a half of sample rate.
    :param nfilter: The number of filter, which ranges canonically from 26 to 40.
    :param nfft: The number of fft we previously used.
    :param sample_rate: The sample rate of the signal
    :return: A matrix containing filter banks.
    """
    low_freq_mel = 2595 * (np.log10(low_freq / 700 + 1))
    high_freq_mel = 2595 * (np.log10(high_freq / 700 + 1))
    arr_freq_mel = np.linspace(low_freq_mel, high_freq_mel, nfilter + 2)
    arr_freq = 700 * (10 ** (arr_freq_mel / 2595) - 1)
    f = np.floor((nfft + 1) * arr_freq / sample_rate)
    f_bank = np.zeros((nfilter, int(nfft / 2) + 1))
    for m in range(1, nfilter + 1):
        left_point = f[m - 1]
        mid_point = f[m]
        right_point = f[m + 1]
        for k in range(int(left_point), int(mid_point)):  # filling the left part
            f_bank[m - 1, k] = (k - f[m - 1]) / (f[m] - f[m - 1])
        for k in range(int(mid_point), int(right_point)):
            f_bank[m - 1, k] = (f[m + 1] - k) / (f[m + 1] - f[m])

    return f_bank


def dynamic_featurization(mfccs: np.ndarray, windowed: np.ndarray) -> np.ndarray:
    r'''
    Do dynamic featurizations, use dct conclusions to combine original features and original total energies,
    and one-stage difference and two-stage difference.
    Suppose that the original feature is 12-dim, so that the total original feature contains 12 + 1(energy) dim.
    And with one, two-stage difference, the total features is 39.
    :param mfccs: The features which was produced by dct.
    :param windowed: Previously windowed signals.
    :return: Features
    '''
    energy = windowed.sum(1)  # type: np.ndarray
    energy = energy.reshape((len(energy), 1))
    # print(mfccs.shape)
    features = np.concatenate([mfccs, energy], 1)

    delta_mfccs = np.concatenate([[mfccs[0]], (mfccs[2:] - mfccs[:-2]) / 2, [mfccs[-1]]])
    delta_energy = np.concatenate([[energy[0]], (energy[2:] - energy[:-2]) / 2, [energy[-1]]])
    features = np.concatenate([features, delta_mfccs, delta_energy], 1)

    delta_delta_mfccs = np.concatenate([[delta_mfccs[0]], (delta_mfccs[2:] - delta_mfccs[:-2]) / 2, [delta_mfccs[-1]]])
    delta_delta_energy = np.concatenate(
        [[delta_energy[0]], (delta_energy[2:] - delta_energy[:-2]) / 2, [delta_energy[-1]]])
    features = np.concatenate([features, delta_delta_mfccs, delta_delta_energy], 1)

    return features


def feature_transform(features: np.ndarray):
    r"""
    Use normalizations, to make the features submit to normal distribution.
    :param features: Features proceeded
    :return: Normalized features.
    """
    mean = features.mean(0)
    std = features.std(0)
    features = (features - mean) / std
    return features


if __name__ == '__main__':
    sig, sample_rate = read_wav('Jiading.wav')

    plt.figure(figsize=(7.5, 15), dpi=200)
    plt.subplot(5, 1, 1)
    plt.plot(sig)
    plt.title('Raw Signal')
    plt.xlabel('Frame')
    plt.ylabel('Power')

    pre_emphasized_signal = pre_emphasis(sig)
    plt.subplot(5, 1, 2)
    plt.plot(pre_emphasized_signal)
    plt.title('Pre Emphasized Signal')
    plt.xlabel('Frame')
    plt.ylabel('Power')
    # plt.show()
    frame = framing(sig, sample_rate)
    windowed = window(frame, frame.shape[1])
    plt.subplot(5, 1, 3)
    plt.imshow(windowed)
    plt.xlabel('Frame')
    plt.ylabel('Window')
    plt.title('Windowed Features')
    # FFT
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(windowed, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    fbanks = mel_filter_bank(low_freq=0, high_freq=int(sample_rate / 2), nfilter=20, nfft=NFFT, sample_rate=sample_rate)
    filter_banks = np.dot(pow_frames, fbanks.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # log part
    filter_banks = np.log10(filter_banks)
    plt.subplot(5, 1, 4)

    plt.imshow(filter_banks[4:, :].T)
    plt.xlabel('Frame')
    plt.ylabel('Window')
    plt.title('Filtered Features')
    # plt.show()
    # go thru dct, here we're using the top 12 dims as the feature, where F0 info is removed
    num_ceps = 12
    mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]
    dynamic_feature = dynamic_featurization(mfccs, windowed)
    features = feature_transform(dynamic_feature)
    plt.subplot(5, 1, 5)

    plt.imshow(features[4:, :].T)
    plt.xlabel('Frame')
    plt.ylabel('Window')
    plt.title('Features')
    plt.subplots_adjust(hspace=0.4)

    plt.show()
