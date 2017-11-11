#!/usr/bin/env python
from io import BytesIO

from pocketsphinx.pocketsphinx import *


class MyPocketSphinx:
    confidence_strategy = 'default'

    def __init__(self, config):
        # Validate and prepare config.
        self._validate_config(config)

        # Create a decoder with certain model.
        ps_config = Decoder.default_config()

        # Disable logging.
        if not config['verbose']:
            ps_config.set_string('-logfn', '/dev/null')

        # Configure PocketSphinx decoder.
        for name, value in config['decoder'].items():
            ps_config.set_string(name, value)

        # Decode streaming data.
        self.decoder = Decoder(ps_config)
        self.set_confidence_strategy(config['confidence_strategy'])
        self.buffer_size = config['buffer_size']
        self.config = config

    @staticmethod
    def _validate_config(config):
        assert 'decoder' in config \
               and '-hmm' in config['decoder'] \
               and '-lm' in config['decoder'] \
               and '-dict' in config['decoder']
        config['buffer_size'] = int(config['buffer_size']) if 'buffer_size' in config else 1024
        config['verbose'] = bool(config['verbose']) if 'verbose' in config else False
        config['confidence_strategy'] = config['confidence_strategy'] if 'confidence_strategy' in config else 'default'

    def set_confidence_strategy(self, confidence_strategy_type):
        known_strategies = ['default', 'by_word']
        if confidence_strategy_type in known_strategies:
            self.confidence_strategy = confidence_strategy_type
        else:
            raise ValueError('Unknown confidence strategy type')

    def transcribe(self, content):
        # Process audio data with PocketSphinx.
        self.decoder.start_utt()
        stream = BytesIO(content)
        while True:
            buf = stream.read(self.buffer_size)
            if buf:
                self.decoder.process_raw(buf, False, False)
            else:
                break
        self.decoder.end_utt()
        self.decoder.get_in_speech()
        # Collect alternatives and confidence scores.
        hypothesis = self.decoder.hyp()
        alternatives = [{
            'service_name': self.config['service_name'],
            'confidence': self.get_confidence(hypothesis),
            'transcript': hypothesis.hypstr
        }]

        return alternatives

    def get_confidence(self, hypothesis):
        if self.confidence_strategy == 'default':
            return self.get_confidence_default(hypothesis)
        elif self.confidence_strategy == 'by_word':
            return self.get_confidence_by_word(hypothesis)
        else:
            return 0

    def get_confidence_default(self, hypothesis):
        logmath = self.decoder.get_logmath()
        return logmath.exp(hypothesis.prob)

    def get_confidence_by_word(self, hypothesis):
        logmath = self.decoder.get_logmath()
        confidence_sum = 0
        count = 0
        for seg in self.decoder.seg():
            if seg.word in hypothesis.hypstr or seg.word == '<sil>':
                confidence_sum = confidence_sum + logmath.exp(seg.prob)
                count = count + 1

        return confidence_sum / count if count else 0
