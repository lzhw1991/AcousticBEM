# ---------------------------------------------------------------------------
# Copyright (C) 2017 Frank Jargstorff
#
# This file is part of the AcousticBEM library.
# AcousticBEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AcousticBEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
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


