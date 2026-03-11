"""Planck function implementation (adapted from DisORT)."""

import numpy as np

PI = np.pi
C2 = 1.438786
SIGMA = 5.67032e-8
VCUT = 1.5
A1 = 1.0 / 3.0
A2 = -1.0 / 8.0
A3 = 1.0 / 60.0
A4 = -1.0 / 5040.0
A5 = 1.0 / 272160.0
A6 = -1.0 / 13305600.0

VCP = [10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0]


def _planck_kernel(v):
    return v * v * v / (np.exp(v) - 1.0)


def planck_function(wnumlo, wnumhi, temp):
    """Compute Planck function integrated between two wavenumbers.

    Args:
        wnumlo: Lower wavenumber [cm^-1].
        wnumhi: Upper wavenumber [cm^-1].
        temp: Temperature [K].

    Returns:
        Integrated Planck function [W/m^2].
    """
    if temp < 0.0 or wnumhi < wnumlo or wnumlo < 0.0:
        raise ValueError("planck_function: invalid arguments")

    if temp < 1.0e-4:
        return 0.0

    if wnumhi == wnumlo:
        arg = np.exp(-C2 * wnumhi / temp)
        return 1.1911e-8 * wnumhi**3 * arg / (1.0 - arg)

    vmax = np.log(np.finfo(np.float64).max)
    sigdpi = SIGMA / PI
    conc = 15.0 / PI**4

    v = [C2 * wnumlo / temp, C2 * wnumhi / temp]

    if (v[0] > np.finfo(np.float64).eps and v[1] < vmax
            and (wnumhi - wnumlo) / wnumhi < 1.0e-2):
        hh = v[1] - v[0]
        oldval = 0.0
        val0 = _planck_kernel(v[0]) + _planck_kernel(v[1])

        for n in range(1, 11):
            dl = hh / (2.0 * n)
            val = val0
            for k in range(1, 2 * n):
                val += 2.0 * (1 + k % 2) * _planck_kernel(v[0] + k * dl)
            val *= dl * A1
            if abs((val - oldval) / val) <= 1.0e-6:
                return sigdpi * temp**4 * conc * val
            oldval = val

        return sigdpi * temp**4 * conc * oldval

    p = [0.0, 0.0]
    d = [0.0, 0.0]
    smallv = 0

    for i in range(2):
        if v[i] < VCUT:
            smallv += 1
            vsq = v[i] * v[i]
            p[i] = conc * vsq * v[i] * (
                A1 + v[i] * (A2 + v[i] * (A3 + vsq * (A4 + vsq * (A5 + vsq * A6))))
            )
        else:
            mmax = 1
            while v[i] < VCP[mmax - 1]:
                mmax += 1
            ex = np.exp(-v[i])
            exm = 1.0
            d[i] = 0.0
            for m in range(1, mmax + 1):
                mv = m * v[i]
                exm *= ex
                d[i] += exm * (6.0 + mv * (6.0 + mv * (3.0 + mv))) / (m**4)
            d[i] *= conc

    if smallv == 2:
        ans = p[1] - p[0]
    elif smallv == 1:
        ans = 1.0 - p[0] - d[1]
    else:
        ans = d[0] - d[1]

    ans *= sigdpi * temp**4

    return ans if ans != 0.0 else 0.0
