import signal

from GoogleCloudSpeechAPI import GoogleCloudSpeechAPI
from MyPocketSphinx import MyPocketSphinx
from VoiceRecord import VoiceRecord
from snowboy import snowboydecoder

interrupted = False
LANGUAGE_CODE = 'ru-RU'


def signal_handler(signal, frame):
    global interrupted
    interrupted = True


def interrupt_callback():
    global interrupted
    return interrupted


def command_handler(voice_record, local_services, cloud_services):
    confidence_threshold = 0.75
    # Listen audio data.
    audio_threshold = 1000
    speech_data = voice_record.get_speech_data(audio_threshold, 1)
    # Concatenate all phrases.
    content = ''.join(speech_data)

    local_alternatives = []
    # Recognize actions locally.
    for local_service in local_services:
        local_alternatives += local_service.transcribe(content)

    for alternative in local_alternatives:
        print(alternative['transcript'], ' ', alternative['confidence'])

    # TODO: Send list of desired commands.
    cloud_alternatives = []
    for cloud_service in cloud_services:
        cloud_alternatives += cloud_service.transcribe(content)

    # TODO: Filter alternatives we don't know.


def main():
    model = "resources/snowboy/snowboy.umdl"

    print('Initializing...')

    # capture SIGINT signal, e.g., Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    detector = snowboydecoder.HotwordDetector(model, sensitivity=0.6)

    voice_record = VoiceRecord(audio_format=detector.stream_in._format,
                               channels=detector.stream_in._channels,
                               rate=detector.stream_in._rate,
                               frames_per_buffer=detector.stream_in._frames_per_buffer)

    gcsr = GoogleCloudSpeechAPI(voice_record.ENCODING, voice_record._rate, 'ru-RU')
    ps_rec = MyPocketSphinx(confidence_strategy='default', verbose=False)

    # main loop
    print('Listening... Press Ctrl+C to exit')
    detector.start(detected_callback=lambda: command_handler(voice_record=voice_record, local_services=[ps_rec],
                                                             cloud_services=[gcsr]),
                   interrupt_check=interrupt_callback,
                   sleep_time=0.001)

    detector.terminate()


if __name__ == '__main__':
    main()
