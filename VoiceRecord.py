import pyaudio
from collections import deque
import numpy


class VoiceRecord:
    ENCODING = 'LINEAR16'

    # Silence limit in seconds. The max amount of seconds where
    # only silence is recorded. When this time passes the
    # recording finishes and the file is delivered.
    SILENCE_LIMIT = 1

    # Silence stop limit in seconds.
    # The max amount of silence seconds waiting.
    SILENCE_STOP_LIMIT = 5
    RECORDING_STOP_LIMIT = 59

    # Previous audio (in seconds) to prepend. When noise
    # is detected, how much of previously recorded audio is
    # prepended. This helps to prevent chopping the beginning
    # of the phrase.
    PREV_AUDIO = 0.5

    def __init__(self, threshold, audio_stream_config):
        self.audio = pyaudio.PyAudio()
        self.threshold = threshold if threshold else 0
        self.stream_in = None
        self.vad = None
        self.verbose = True
        # Set format configs for PyAudio audio stream.
        self._audio_format = audio_stream_config['format']
        self._channels = audio_stream_config['channels']
        self._rate = audio_stream_config['rate']
        self._chunk = audio_stream_config['frames_per_buffer']  # CHUNKS of bytes to read each time from mic

    def stream_open(self):
        """
        Opens audio stream.
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
        self.log("Getting intensity values from mic.")

        stream = self.stream_open()

        values = [self.get_vad_estimate(stream.read(self._chunk)) for _ in range(num_samples)]
        values = sorted(values, reverse=True)
        r = sum(values[:int(num_samples * 0.2)]) / int(num_samples * 0.2)

        stream.close()

        self.log(" Finished ")
        self.log(" Average audio intensity is ", str(r))

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
        if threshold is None:
            threshold = self.threshold

        self.log("* Listening mic. ")
        recorded_phrase = []
        rel = int(self._rate / self._chunk)
        slid_win_maxlen = int(self.SILENCE_LIMIT * rel)
        prev_audio_maxlen = int(self.PREV_AUDIO * rel)
        slid_win = deque(maxlen=slid_win_maxlen)

        # Calculate recorded size in bytes for limits.
        silence_stop_len = int(self.SILENCE_STOP_LIMIT * rel)
        recording_stop_len = int(self.RECORDING_STOP_LIMIT * rel)

        # Prepend audio from 0.5 seconds before noise was detected
        prev_audio = deque(maxlen=prev_audio_maxlen)

        # Set initial recording values.
        started = False
        n = num_phrases
        speech_data = []
        recorded_chunks = 0

        while num_phrases == -1 or n > 0:
            # Current chunk of audio data.
            cur_data = stream.read(self._chunk)
            recorded_chunks += 1
            # TODO: Fix VAD estimation. It starts without voice.
            slid_win.append(self.get_vad_estimate(cur_data))
            threshold_pass_num = sum([x >= threshold for x in slid_win])

            # Check if we are over the allowed limit.
            recording_limit_passed = recorded_chunks > recording_stop_len

            # If it's not some random peak, don't start recording.
            if threshold_pass_num > 1 and not recording_limit_passed:
                if not started:
                    self.log("Starting recording of a phrase")
                    started = True
                recorded_phrase.append(cur_data)
                recorded_chunks = len(recorded_phrase)

            elif started is True:
                self.log("Finished")
                if recording_limit_passed:
                    self.log("Stopped recording over {} seconds".format(self.RECORDING_STOP_LIMIT))

                # The limit was reached, finish capture and deliver.
                speech_data.append(b''.join(list(prev_audio) + recorded_phrase))
                # Reset all.
                started = False
                recorded_chunks = 0
                slid_win = deque(maxlen=slid_win_maxlen)
                prev_audio = deque(maxlen=prev_audio_maxlen)
                recorded_phrase = []
                n -= 1
                if n > 0 or n == -1:
                    self.log("Listening ...")
            else:
                # Exit loop on long silence.
                if recorded_chunks > silence_stop_len:
                    self.log("Stop on long silence")
                    break
                else:
                    prev_audio.append(cur_data)

        self.log("* Done recording")
        stream.close()

        return speech_data

    def get_vad_estimate(self, data):
        if self.vad:
            numpydata = self.bytestring_to_numpy_array(data)
            return self.vad.estimate(numpydata)

        return 1

    def log(self, *args):
        if self.verbose:
            print(' '.join(args))

    def bytestring_to_numpy_array(self, data):
        """
        Converts audio input byte string to numpy array.
        :param data: audio byte string
        :return: numpy representation
        """
        dtype = self.get_numpy_type_for_audio_format(self._audio_format)
        return numpy.fromstring(data, dtype=dtype)

    def get_numpy_type_for_audio_format(self, audio_format):
        """
        Matches pyaudio format to numpy variable type.
        :param audio_format: pyaudio audio format
        :return: numpy format
        """
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
