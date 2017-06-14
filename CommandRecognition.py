import multiprocessing
import sys
from multiprocessing import Process
from GoogleCloudSpeechAPI import GoogleCloudSpeechAPI
from MyPocketSphinx import MyPocketSphinx
from VoiceRecord import VoiceRecord

if sys.version_info > (3, 0):
    from snowboy_python3 import snowboydecoder
else:
    from snowboy_python import snowboydecoder


class CommandRecognition(Process):
    LANGUAGE_CODE = 'ru-RU'

    def command_handler(self, local_services, cloud_services):
        confidence_threshold = 0.2
        # Listen audio data.
        speech_data = self.voice_record.get_speech_data(num_phrases=1)
        if not speech_data:
            # Nothing to do, nothing was caught.
            return
        # Concatenate all phrases.
        content = b''.join(speech_data)

        local_alternatives = []
        # Recognize actions locally.
        for local_service in local_services:
            local_alternatives += local_service.transcribe(content)

        # TODO: Send list of desired commands.
        cloud_alternatives = []
        for cloud_service in cloud_services:
            cloud_alternatives += cloud_service.transcribe(content)
        # TODO: Filter alternatives we don't know.
        self.last_result = cloud_alternatives + local_alternatives

        print("Recognition alternatives")

        # Notify the subscribers.
        self.notify_result(self.last_result)

    def interrupt_callback(self):
        return bool(self.interrupted.value)

    def __init__(self, transport):
        Process.__init__(self)
        self.interrupted = multiprocessing.Value('i', False)
        self.detector = None
        self.voice_record = None
        self.transport = transport
        self.last_result = []

    def run(self):
        self.start_recognize_loop()

    def stop_process(self):
        self.stop_recognize_loop()
        self.join()
        self.terminate()

    def start_recognize_loop(self):
        model = "resources/snowboy/snowboy.umdl"

        print('Initializing...')

        self.detector = snowboydecoder.HotwordDetector(model, sensitivity=0.6)

        self.voice_record = VoiceRecord(threshold=0,
                                        audio_format=self.detector.stream_in._format,
                                        channels=self.detector.stream_in._channels,
                                        rate=self.detector.stream_in._rate,
                                        frames_per_buffer=self.detector.stream_in._frames_per_buffer)
        self.voice_record.set_vad_type('wavelet')
        self.voice_record.threshold = self.voice_record.measure_background_noise(num_samples=20)

        gcsr = GoogleCloudSpeechAPI(self.voice_record.ENCODING, self.voice_record._rate, 'ru-RU')
        ps_rec = MyPocketSphinx(confidence_strategy='default', verbose=False)

        # main loop
        print('Listening...')
        self.set_interrupted(False)
        self.detector.start(
            detected_callback=lambda: self.command_handler(local_services=[ps_rec], cloud_services=[gcsr]),
            interrupt_check=self.interrupt_callback,
            sleep_time=0.001)

        print('Stop listening')
        self.detector.terminate()

    def stop_recognize_loop(self):
        self.set_interrupted(True)

    def set_interrupted(self, value):
        self.interrupted.value = bool(value)

    def notify_result(self, result):
        print('Notifying parent process')
        self.transport.send(result)
