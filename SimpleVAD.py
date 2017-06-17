import numpy


class SimpleVAD:
    def estimate(self, data):
        assert type(data) is numpy.ndarray
        return numpy.sqrt(numpy.absolute(data.mean()))