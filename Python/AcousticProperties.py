import numpy as np

def wavenumberToFrequency(k, c = 344.0):
    return 0.5 * k * c / np.pi

def frequencyToWavenumber(f, c = 344.0):
    return 2.0 * np.pi * f / c

def soundPressure(k, phi, t = 0.0, c = 344.0, density = 1.205):
    angularVelocity = k * c
    return (1j * density * angularVelocity  * np.exp(-1.0j*angularVelocity*t)
            * phi).astype(np.complex64)

def SoundMagnitude(pressure):
    return np.log10(np.abs(pressure / 2e-5)) * 20

def AcousticIntensity(pressure, velocity):
    return 0.5 * (np.conj(pressure) * velocity).real

def SignalPhase(pressure):
    return np.arctan2(pressure.imag, pressure.real)


