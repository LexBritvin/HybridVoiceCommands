import numpy
import pywt


class WaveletVAD:
    def __init__(self, wavelet_type='db4', layer_level=3):
        self.wavelet_type = wavelet_type
        self.layer_level = layer_level

    def estimate(self, data):
        subbands = []

        data_to_process = data
        while len(subbands) < self.layer_level:
            cA, cD = pywt.dwt(data_to_process, self.wavelet_type)
            subbands.append(cD)
            data_to_process = cA
        # Add the last appropriated scale A.
        subbands.append(data_to_process)

        t = [self.teo(s) for s in subbands]
        sae = numpy.float64()
        for ts in t:
            if len(ts):
                acf = self.acf(ts)
                sae += self.mdsacf(acf)

        return sae

    def teo(self, x):
        """ Applies Teager Kaiser energy operator to dataset. """
        # TEO(X[n]) = X[n]^2 - X[n+1] * X[n-1]
        return numpy.multiply(x[1:-1], x[1:-1]) - numpy.multiply(x[2:], x[0:-2])

    def mean_operator(self, x):
        assert type(x) is numpy.ndarray
        return numpy.absolute(x).mean()

    def acf(self, x):
        """ Auto-Correlation function. """
        return numpy.correlate(x, x, mode='full')[-len(x):]

    def mdsacf(self, acf, m=3):
        """
        Mean-Delta Subband Auto-Correlation Function (MDSACF)

        Parameters
        ----------
        acf : array_like
            Auto-Correlation function data.
        m : number
            M-sample neighborhood (lag)

        Returns
        -------
        mdsacf :
            ndarray
            Returns MDSACF.
        """
        assert type(acf) is numpy.ndarray
        n = len(acf)
        # Precalculate R0 and squared sum for M range.
        R0 = acf[0]
        # Arrange M for further calculations.
        mvals = numpy.arange(-m, m + 1, 1, dtype=numpy.float64)

        # Calculate Delta Subband Auto-Correlation Function (DSACF).
        Rm = numpy.zeros(n, dtype=numpy.float64)
        for k, val in enumerate(Rm):
            # Generate ACF(k+m) vector.
            Rk = mvals.copy()
            for ri, rm in enumerate(Rk):
                i = int(k + rm)
                Rk[ri] = acf[i] if 0 <= i < n else numpy.float64(0.0)
            # Generate DSACF vector.
            Rm[k] = (mvals.copy() * Rk / R0).sum()
        # Calculate DSACF.
        Rm /= numpy.square(mvals).sum()

        # Calculate Mean-Delta over Delta Subband Auto-Correlation Function
        return self.mean_operator(Rm)
