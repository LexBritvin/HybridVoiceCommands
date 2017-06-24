import sys
from CommandRecognition import CommandRecognition

if sys.version_info > (3, 0):
    from PyQt5 import QtCore
else:
    from PyQt4 import QtCore

from multiprocessing import Pipe
from threading import Thread


class CommandHandler(QtCore.QObject):
    def __init__(self, receiver):
        super(CommandHandler, self).__init__()
        self.receiver = receiver
        self.receiver.start()
        self.n = 0
        self.receiver.received_signal.connect(self.print_alternatives)

    def print_alternatives(self, alternatives):
        print("Handler alternatives: ")
        for alternative in alternatives:
            print(alternative['transcript'], ': ', alternative['confidence'])
        self.n += 1
        if self.n >= 3:
            QtCore.QCoreApplication.instance().quit()


class CommandReceiver(QtCore.QObject, Thread):
    received_signal = QtCore.pyqtSignal('PyQt_PyObject')

    def __init__(self, transport):
        QtCore.QObject.__init__(self)
        Thread.__init__(self)
        self.transport = transport

    def _emit(self, alternatives):
        self.received_signal.emit(alternatives)

    def run(self):
        while True:
            try:
                alternatives = self.transport.recv()
            except EOFError:
                # TODO: Shutdown gracefully.
                break
            else:
                print("Emitter received")
                self._emit(alternatives)


def main():
    # Create 2 ends of a pipe for communication.
    mother_pipe, child_pipe = Pipe()
    receiver = CommandReceiver(mother_pipe)

    recognition = CommandRecognition(child_pipe)
    recognition.start()

    app = QtCore.QCoreApplication(sys.argv)
    # Received commands handler. Saving a variable is required for Qt to work.
    handler = CommandHandler(receiver)
    app.exec_()

    print('Stopped waiting')
    recognition.stop_process()


if __name__ == '__main__':
    main()
