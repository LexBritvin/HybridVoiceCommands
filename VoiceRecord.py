import pyaudio
import audioop
from collections import deque
import math
import pywt
import numpy


class VoiceRecord:
    ENCODING = 'LINEAR16'

    # Silence limit in seconds. The max amount of seconds where
    # only silence is recorded. When this time passes the
    # recording finishes and the file is delivered.
    SILENCE_LIMIT = 1

    # Previous audio (in seconds) to prepend. When noise
    # is detected, how much of previously recorded audio is
    # prepended. This helps to prevent chopping the beginning
    # of the phrase.
    PREV_AUDIO = 0.5

    def __init__(self, threshold, audio_format, channels, rate, frames_per_buffer):
        self.audio = pyaudio.PyAudio()
        self.stream_in = None
        self.threshold = threshold
        self.vad_type = 'default'
        self._audio_format = audio_format
        self._channels = channels
        self._rate = rate
        self._chunk = frames_per_buffer  # CHUNKS of bytes to read each time from mic

    def stream_open(self):
        """
        TODO: Description
        :return:
        """
        self.stream_in = self.audio.open(format=self._audio_format,
                                         channels=self._channels,
                                         rate=self._rate,
                                         input=True,
                                         frames_per_buffer=self._chunk)
        return self.stream_in

    def measure_background_noise(self, num_samples=50):
        """
        Gets average audio intensity of your mic sound. You can use it to get
        average intensities while you're talking and/or silent. The average
        is the avg of the 20% largest intensities recorded.
        """
        stream = self.stream_open()

        print("Getting intensity values from mic.")
        values = [self.get_vad_estimate(stream.read(self._chunk)) for _ in range(num_samples)]
        values = sorted(values, reverse=True)
        r = sum(values[:int(num_samples * 0.2)]) / int(num_samples * 0.2)
        print(" Finished ")
        print(" Average audio intensity is ", r)
        stream.close()
        return r

    def get_speech_data(self, threshold=None, num_phrases=-1):
        """
        Listens to Microphone, extracts phrases from it. A "phrase" is sound
        surrounded by silence (according to threshold). num_phrases controls
        how many phrases to process before finishing the listening process
        (-1 for infinite).
        """

        # Open stream
        stream = self.stream_open()

        print("* Listening mic. ")
        recorded_phrase = []
        rel = int(self._rate / self._chunk)
        slid_win_maxlen = int(self.SILENCE_LIMIT * rel)
        prev_audio_maxlen = int(self.PREV_AUDIO * rel)
        slid_win = deque(maxlen=slid_win_maxlen)
        # Prepend audio from 0.5 seconds before noise was detected
        prev_audio = deque(maxlen=prev_audio_maxlen)
        started = False
        n = num_phrases
        speech_data = []

        while num_phrases == -1 or n > 0:
            # Current chunk of audio data.
            cur_data = stream.read(self._chunk)
            slid_win.append(self.get_vad_estimate(cur_data))

            if sum([x > threshold for x in slid_win]) > 0:
                if not started:
                    print("Starting record of phrase")
                    started = True
                # TODO: Prevent recording longer than X sec.
                recorded_phrase.append(cur_data)
            elif started is True:
                print("Finished")
                # The limit was reached, finish capture and deliver.
                speech_data.append(b''.join(list(prev_audio) + recorded_phrase))
                # Reset all
                started = False
                slid_win = deque(maxlen=slid_win_maxlen)
                prev_audio = deque(maxlen=prev_audio_maxlen)
                recorded_phrase = []
                n -= 1
                if n > 0 or n == -1:
                    print("Listening ...")
            else:
                # TODO: Add exit from loop on long silence.
                prev_audio.append(cur_data)

        print("* Done recording")
        stream.close()

        return speech_data

    def set_vad_type(self, vad_type):
        known_types = ['default', 'wavelet']
        if vad_type in known_types:
            self.vad_type = vad_type
        else:
            raise ValueError('Unknown VAD type')

    def get_vad_estimate(self, data):
        if self.vad_type == 'default':
            return self.default_vad_estimate(data)
        elif self.vad_type == 'wavelet':
            numpydata = self.bytestring_to_numpy_array(data)
            return self.wavelet_vad_estimate(numpydata)

        return 0

    def default_vad_estimate(self, data):
        return math.sqrt(abs(audioop.avg(data, 4)))

    def wavelet_vad_estimate(self, data):
        wavelet = 'db4'
        level = 3
        subbands = []

        data_to_process = data
        while len(subbands) < level:
            cA, cD = pywt.dwt(data_to_process, wavelet)
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

    def get_numpy_type_for_audio_format(self, audio_format):
        known_formats = {
            pyaudio.paUInt8: numpy.uint8,
            pyaudio.paInt8: numpy.int8,
            pyaudio.paInt16: numpy.int16,
            pyaudio.paInt32: numpy.int32,
            pyaudio.paFloat32: numpy.float32
        }
        if audio_format in known_formats:
            return known_formats[audio_format]

        raise ValueError('Unhandled audio format', audio_format)

    def bytestring_to_numpy_array(self, data):
        dtype = self.get_numpy_type_for_audio_format(self._audio_format)
        return numpy.fromstring(data, dtype=dtype)
