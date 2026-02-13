# wavelet_functions.py
import sys

import numpy as np
import pandas as pd
# from scipy.signal import convolve
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
# from scipy.signal.windows import hilbert


from scipy import signal
from scipy.linalg import toeplitz
from scipy.fft import fft, ifft



def objective_function(wavelet, seismic_trace, reflectivity):
    synthetic = np.convolve(reflectivity, wavelet, mode='same')
    return np.sum((synthetic - seismic_trace) ** 2)
def getFullWavelet(wavelet):
    return np.append(wavelet[::-1],wavelet)
# Least Squares Wavelet Estimation or Deterministic Wavelet Extraction using optimization

def roy_white_method(seismic_trace, reflectivity, wavelet_length=64, epsilon=1e-6):
    # Compute autocorrelation of reflectivity
    phi_rr = signal.correlate(reflectivity, reflectivity, mode='full', method='fft')
    phi_rr = phi_rr[phi_rr.size//2:][:wavelet_length]
    
    # Compute cross-correlation of seismic trace and reflectivity
    phi_sr = signal.correlate(seismic_trace, reflectivity, mode='full', method='fft')
    phi_sr = phi_sr[phi_sr.size//2:][:wavelet_length]
    
    # Compute wavelet spectrum
    Phi_rr = fft(phi_rr)
    Phi_sr = fft(phi_sr)
    W = Phi_sr / (Phi_rr + epsilon)
    
    # Transform back to time domain
    w_est = np.real(ifft(W))
    
    # Apply a taper
    taper = signal.tukey(wavelet_length, alpha=0.1)
    w_est *= taper
    
    return w_est

def constraints_inversion_method(seismic_trace, reflectivity, wavelet_length=64, lambda1=0.1, lambda2=0.1):
    N = len(seismic_trace)
    M = len(reflectivity)
    
    # Create the Toeplitz matrix R with appropriate dimensions
    R = toeplitz(np.pad(reflectivity, (0, wavelet_length - 1), 'constant'), np.zeros(N))
    
    # Create smoothing operator
    L = np.diff(np.eye(wavelet_length), n=2, axis=0)
    
    # Create prior wavelet (simple spike in this case)
    w_prior = np.zeros(wavelet_length)
    w_prior[wavelet_length // 2] = 1
    
    # Solve the inverse problem
    RtR = R[:N, :wavelet_length].T @ R[:N, :wavelet_length]  # Ensuring dimensions match
    A = RtR + lambda1 * L.T @ L + lambda2 * np.eye(wavelet_length)
    b = R[:N, :wavelet_length].T @ seismic_trace + lambda2 * w_prior
    
    w_est = np.linalg.solve(A, b)
    
    return w_est
    
def extract_wavelet_deterministic(seismic_trace, reflectivity, wavelet_length=64, taper_alpha=0.1):
    wavelet_length=wavelet_length//2
    initial_wavelet = np.zeros(wavelet_length)
    initial_wavelet[wavelet_length // 2] = 1
    
    result = minimize(
        objective_function,
        initial_wavelet,
        args=(seismic_trace, reflectivity),
        method='L-BFGS-B'
    )
    
    wavelet = result.x
    wavelet /= np.sqrt(np.sum(wavelet ** 2))  # Energy-preserving normalization
    full_wavelet=getFullWavelet(wavelet)
    # Apply taper
    taper = tukey(len(full_wavelet), alpha=taper_alpha)
    tapered_wavelet = full_wavelet * taper
    
    # Re-normalize after tapering
    tapered_wavelet /= np.sqrt(np.sum(tapered_wavelet ** 2))
    
    return tapered_wavelet



def frequency_domain_wavelet(trace_data, wavelet_length=64):
    # Check if input is 1D or 2D
    if trace_data.ndim == 1:
        # If 1D, reshape to 2D
        trace_data = trace_data.reshape(1, -1)
    
    fft_traces = np.fft.fft(trace_data, axis=1)
    average_spectrum = np.mean(np.abs(fft_traces), axis=0)
    wavelet = np.fft.ifft(average_spectrum)
    return np.real(wavelet)[:wavelet_length]

def extract_wavelet_autocorrelation(trace, wavelet_length):
    autocorr = np.correlate(trace, trace, mode='full')
    center = len(autocorr) // 2
    wavelet = autocorr[center:center+wavelet_length]
    return wavelet / np.max(np.abs(wavelet))

def evaluate_wavelet( actual_seismic, reflectivity, wavelet):
    def create_synthetic_seismic(reflectivity, wavelet):
        return np.convolve(reflectivity, wavelet, mode='same')
    synthetic_seismic = create_synthetic_seismic(reflectivity, wavelet)
    correlation = np.corrcoef(actual_seismic, synthetic_seismic)[0, 1]
    rmse = np.sqrt(np.mean((actual_seismic - synthetic_seismic)**2))
    
    # Spectral analysis
    actual_freq = np.abs(np.fft.fft(actual_seismic))
    synth_freq = np.abs(np.fft.fft(synthetic_seismic))
    spectral_correlation = np.corrcoef(actual_freq, synth_freq)[0, 1]
    
    # Energy distribution
    wavelet_energy = np.sum(wavelet**2)
    central_energy = np.sum(wavelet[len(wavelet)//4:3*len(wavelet)//4]**2)
    energy_ratio = central_energy / wavelet_energy
    
    return {
        'synthetic_seismic':synthetic_seismic,
        'correlation': correlation,
        'rmse': rmse,
        'spectral_correlation': spectral_correlation,
        'energy_ratio': energy_ratio
    }


def ricker_wavelet(f, length, dt):
    """
    Generate a Ricker wavelet.
    
    Parameters:
    f : float
        Peak frequency of the wavelet
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    
    Returns:
    numpy.ndarray: Ricker wavelet
    """
    t = np.arange(length) * dt - (length - 1) * dt / 2
    y = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return y


def ormsby_wavelet(f1, f2, f3, f4, length, dt):
    """
    Generate an Ormsby wavelet.
    
    Parameters:
    f1, f2, f3, f4 : float
        Corner frequencies of the Ormsby wavelet
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    
    Returns:
    numpy.ndarray: Ormsby wavelet
    """
    t = np.arange(length) * dt - (length - 1) * dt / 2
    pi = np.pi
    y = ((pi * f4) ** 2 / (pi * f4 - pi * f3) * np.sinc(f4 * t) ** 2
         - (pi * f3) ** 2 / (pi * f4 - pi * f3) * np.sinc(f3 * t) ** 2
         - (pi * f2) ** 2 / (pi * f2 - pi * f1) * np.sinc(f2 * t) ** 2
         + (pi * f1) ** 2 / (pi * f2 - pi * f1) * np.sinc(f1 * t) ** 2)
    return y

def klauder_wavelet(f1, f2, length, dt, phase=0):
    """
    Generate a Klauder wavelet.
    
    Parameters:
    f1, f2 : float
        Start and end frequencies of the sweep
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    phase : float, optional
        Phase shift in radians (default is 0)
    
    Returns:
    numpy.ndarray: Klauder wavelet
    """
    t = np.arange(length) * dt - (length - 1) * dt / 2
    k = (f2 - f1) / (length * dt)
    y = np.sin(2 * np.pi * (f1 * t + k * t ** 2 / 2) + phase)
    return y

def morlet_wavelet(f, length, dt):
    """
    Generate a Morlet wavelet.
    
    Parameters:
    f : float
        Central frequency of the wavelet
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    
    Returns:
    numpy.ndarray: Morlet wavelet
    """
    t = np.arange(length) * dt - (length - 1) * dt / 2
    y = np.cos(2 * np.pi * f * t) * np.exp(-(t ** 2) / 2)
    return y

def gabor_wavelet(f, sigma, length, dt):
    """
    Generate a Gabor wavelet.
    
    Parameters:
    f : float
        Central frequency of the wavelet
    sigma : float
        Standard deviation of the Gaussian envelope
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    
    Returns:
    numpy.ndarray: Gabor wavelet
    """
    t = np.arange(length) * dt - (length - 1) * dt / 2
    y = np.exp(-(t ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * f * t)
    return y

def berlage_wavelet(f, n, alpha, length, dt):
    """
    Generate a Berlage wavelet.
    
    Parameters:
    f : float
        Dominant frequency
    n : int
        Time exponent
    alpha : float
        Exponential decay factor
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    
    Returns:
    numpy.ndarray: Berlage wavelet
    """
    t = np.arange(length) * dt
    y = (t ** n) * np.exp(-alpha * t) * np.cos(2 * np.pi * f * t)
    return y

def butterworth_wavelet(f, order, length, dt):
    """
    Generate a Butterworth wavelet.
    
    Parameters:
    f : float
        Cut-off frequency
    order : int
        Filter order
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    
    Returns:
    numpy.ndarray: Butterworth wavelet
    """
    nyq = 1 / (2 * dt)
    normal_cutoff = f / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    impulse = np.zeros(length)
    impulse[length // 2] = 1
    y = signal.lfilter(b, a, impulse)
    return y

def gaussian_wavelet(f, length, dt):
    """
    Generate a Gaussian wavelet.
    
    Parameters:
    f : float
        Dominant frequency
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    
    Returns:
    numpy.ndarray: Gaussian wavelet
    """
    t = np.arange(length) * dt - (length - 1) * dt / 2
    sigma = 1 / (2 * np.pi * f)
    y = np.exp(-(t ** 2) / (2 * sigma ** 2))
    return y

def sinc_wavelet(f, length, dt):
    """
    Generate a Sinc wavelet.
    
    Parameters:
    f : float
        Cut-off frequency
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    
    Returns:
    numpy.ndarray: Sinc wavelet
    """
    t = np.arange(length) * dt - (length - 1) * dt / 2
    y = np.sinc(2 * f * t)
    return y

def boxcar_wavelet(width, length, dt):
    """
    Generate a Boxcar wavelet.
    
    Parameters:
    width : float
        Width of the boxcar in seconds
    length : int
        Number of samples
    dt : float
        Sampling interval in seconds
    
    Returns:
    numpy.ndarray: Boxcar wavelet
    """
    t = np.arange(length) * dt - (length - 1) * dt / 2
    y = np.where(np.abs(t) <= width / 2, 1, 0)
    return y

if __name__ == '__main__':
    f=30.0
    length=128
    dt=2.0
    wl=ricker_wavelet(f, length, dt)