#!/usr/bin/python

import google.auth
import google.auth.transport.grpc
import google.auth.transport.requests
from google.cloud.proto.speech.v1beta1 import cloud_speech_pb2

# Keep the request alive for this many seconds
DEADLINE_SECS = 60
SPEECH_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
LANGUAGE_CODE = 'en-US'


class GoogleCloudSpeechAPI:
    """
    TODO: Description.
    """

    def __init__(self, config):
        self._validate_config(config)
        self.encoding = config['audio']['encoding']
        self.sample_rate = config['audio']['rate']
        self.language_code = config['language_code']

    @staticmethod
    def _validate_config(config):
        config['language_code'] = config['language_code'] if 'language_code' in config else LANGUAGE_CODE

    def make_channel(self, host, port):
        """Creates a secure channel with auth credentials from the environment."""
        # Grab application default credentials from the environment
        credentials, _ = google.auth.default(scopes=[SPEECH_SCOPE])

        # Create a secure channel using the credentials.
        http_request = google.auth.transport.requests.Request()
        target = '{}:{}'.format(host, port)

        return google.auth.transport.grpc.secure_authorized_channel(
            credentials, http_request, target)

    def transcribe(self, content):
        return self.request_transcribe_sync(content)

    def request_transcribe_sync(self, content):
        """
        TODO: Description.
        :param content:
        :return:
        """
        service = cloud_speech_pb2.SpeechStub(self.make_channel('speech.googleapis.com', 443))

        # The method and parameters can be inferred from the proto from which the
        # grpc client lib was generated. See:
        # https://github.com/googleapis/googleapis/blob/master/google/cloud/speech/v1beta1/cloud_speech.proto
        response = service.SyncRecognize(cloud_speech_pb2.SyncRecognizeRequest(
            config=cloud_speech_pb2.RecognitionConfig(
                # There are a bunch of config options you can specify. See
                # https://goo.gl/KPZn97 for the full list.
                encoding=self.encoding,  # one of LINEAR16, FLAC, MULAW, AMR, AMR_WB
                sample_rate=self.sample_rate,  # the rate in hertz
                # See https://g.co/cloud/speech/docs/languages for a list of
                # supported languages.
                language_code=self.language_code,  # a BCP-47 language tag
            ),
            audio=cloud_speech_pb2.RecognitionAudio(
                content=content,
            )
        ), DEADLINE_SECS)

        # Print the recognition result alternatives and confidence scores.
        alternatives = []
        for result in response.results:
            for alternative in result.alternatives:
                alternatives.append({
                    'confidence': alternative.confidence,
                    'transcript': alternative.transcript
                })
        return sorted(alternatives, key=lambda k: k['confidence'], reverse=True)
