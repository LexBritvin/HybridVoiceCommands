#!/usr/bin/env python
from io import BytesIO
from os import path

from pocketsphinx.pocketsphinx import *

MODELDIR = "resources/pocketsphinx/model"


class MyPocketSphinx:
    confidence_strategy = 'default'

    def __init__(self, confidence_strategy='default', verbose=False):
        # Create a decoder with certain model
        config = Decoder.default_config()
        config.set_string('-hmm', path.join(MODELDIR, 'ru-ru/cmu_ru-ru'))
        config.set_string('-lm', path.join(MODELDIR, 'ru-ru/robot.lm.bin'))
        config.set_string('-dict', path.join(MODELDIR, 'ru-ru/robot.dic'))

        # Disable logging
        if not verbose:
            config.set_string('-logfn', '/dev/null')

        # Decode streaming data.
        self.decoder = Decoder(config)
        self.set_confidence_strategy(confidence_strategy)

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
            buf = stream.read(1024)
            if buf:
                self.decoder.process_raw(buf, False, False)
            else:
                break
        self.decoder.end_utt()
        self.decoder.get_in_speech()
        # Collect alternatives and confidence scores.
        hypothesis = self.decoder.hyp()
        alternatives = [{
            'confidence': self.get_confidence(hypothesis),
            'transcript': hypothesis.hypstr
        }]

        return sorted(alternatives, key=lambda k: k['confidence'], reverse=True)

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
