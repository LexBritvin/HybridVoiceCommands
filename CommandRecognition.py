import multiprocessing
from multiprocessing import Process, Pipe

from GoogleCloudSpeechAPI import GoogleCloudSpeechAPI
from MyPocketSphinx import MyPocketSphinx
from VoiceRecord import VoiceRecord

from snowboy import snowboydecoder
import yaml


class CommandRecognition(Process):
    LANGUAGE_CODE = 'ru-RU'

    def __init__(self):
        Process.__init__(self)
        # Init interprocess variables.
        self.interrupted = multiprocessing.Value('i', False)

        # Detector configs.
        self.detector = None
        self.voice_record = None
        self.last_result = []

        self.config = {}
        self.local_services = []
        self.cloud_services = []

        # Create transport to send commands.
        self.external_transport = None
        self.transport = None
        self.init_pipe_transport()

    def init_pipe_transport(self):
        # Create 2 ends of a pipe for communication.
        mother_pipe, child_pipe = Pipe()
        self.external_transport = mother_pipe
        self.transport = child_pipe

    def get_external_transport(self):
        return self.external_transport

    def get_pipe(self):
        return self.external_transport, self.transport

    def run(self):
        # Implementation of Process.run().
        self.start_recognize_loop()

    def stop_process(self):
        # Gracefully stops the process.
        self.stop_recognize_loop()
        self.join()
        self.terminate()

    def start_recognize_loop(self):
        print('Initializing...')

        # Configure Hotword detection.
        model = self.config['hotword_detector']['model']
        sensitivity = self.config['hotword_detector']['sensitivity']
        self.detector = snowboydecoder.HotwordDetector(model, sensitivity=sensitivity)

        # Configure voice recorder
        self.config['recorder']['audio'] = self.get_stream_config()
        self.voice_record = VoiceRecord(self.config['recorder'])
        if 'bg_noise_samples' in self.config['recorder']:
            self.voice_record.threshold = self.voice_record.measure_background_noise(num_samples=self.config['recorder']['bg_noise_samples'])

        # Store audio config for services.
        self.config['audio'] = self.config['recorder']['audio']
        self.config['audio']['encoding'] = self.voice_record.ENCODING

        self.create_services()

        # main loop
        print('Listening...')
        self.set_interrupted(False)
        self.detector.start(
            detected_callback=lambda: self.command_handler(),
            interrupt_check=self.interrupt_callback,
            sleep_time=0.001)

        print('Stop listening')
        self.detector.terminate()

    def create_services(self):
        if 'services' not in self.config:
            return

        for service_config in self.config['services']:
            # Pass audio config also.
            service_config['audio'] = self.config['audio']

            # Init service class.
            service = self.init_service(service_config)
            is_local = 'local' in service_config and service_config['local']

            if not service:
                continue

            if is_local:
                self.local_services.append(service)
            else:
                self.cloud_services.append(service)

    @staticmethod
    def init_service(config):
        service = None
        if config['service_name'] == 'pocketsphinx':
            service = MyPocketSphinx(config)
        elif config['service_name'] == 'google':
            service = GoogleCloudSpeechAPI(config)

        return service

    def command_handler(self):
        confidence_threshold = self.config['handler_behaviour']['confidence_threshold']

        # Listen audio data.
        speech_data = self.voice_record.get_speech_data(num_phrases=1)
        if not speech_data:
            # Nothing to do, nothing was caught.
            return
        # Concatenate all phrases.
        content = b''.join(speech_data)

        local_alternatives = []
        # Recognize actions locally.
        for local_service in self.local_services:
            local_alternatives += local_service.transcribe(content)

        max_local_confidence = 0
        for alt in local_alternatives:
            if alt['confidence'] > max_local_confidence:
                max_local_confidence = alt['confidence']

        cloud_alternatives = []
        # Send to cloud if confidence is low.
        # if max_local_confidence < confidence_threshold:
        #     for cloud_service in self.cloud_services:
        #         cloud_alternatives += cloud_service.transcribe(content)

        # Merge all results.
        self.last_result = cloud_alternatives + local_alternatives
        self.last_result = sorted(self.last_result, key=lambda k: k['confidence'], reverse=True)

        # Notify the subscribers.
        self.notify_result(self.last_result)

    def notify_result(self, result):
        print('Notifying parent process')
        self.transport.send(result)

    def interrupt_callback(self):
        # Callback to check current state of interrupted flag.
        # Accesses interprocess variable.
        return bool(self.interrupted.value)

    def stop_recognize_loop(self):
        self.set_interrupted(True)

    def set_interrupted(self, value):
        self.interrupted.value = bool(value)

    def get_stream_config(self):
        if not self.detector.stream_in:
            return {}

        return {
            'format': self.detector.stream_in._format,
            'channels': self.detector.stream_in._channels,
            'rate': self.detector.stream_in._rate,
            'frames_per_buffer': self.detector.stream_in._frames_per_buffer
        }

    def set_config_yaml(self, filepath):
        with open(filepath, 'r') as stream:
            config = yaml.load(stream)
            self.set_config(config)

    def set_config(self, config):
        assert config['hotword_detector']['service_name'] == 'snowboy'
        assert type(config['hotword_detector']['model']) is str
        assert type(config['hotword_detector']['sensitivity']) is float \
            or type(config['hotword_detector']['sensitivity']) is int
        assert type(config['handler_behaviour']['confidence_threshold']) is float
        self.config = config
