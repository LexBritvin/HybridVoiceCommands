from CommandRecognition import CommandRecognition
from multiprocessing import Pipe



class CommandHandler:
    def __init__(self, transport):
        self.transport = transport
        self.n = 0

    def dummy_loop(self):
        while self.n < 3:
            print('Waiting for command')
            alternatives = self.transport.recv()
            self.print_alternatives(alternatives)
            self.n += 1

    def print_alternatives(self, alternatives):
        print("Handler alternatives: ")
        for alternative in alternatives:
            print(alternative['transcript'], ': ', alternative['confidence'])


def main():
    # Create 2 ends of a pipe for communication.
    mother_pipe, child_pipe = Pipe()

    # Received commands handler.
    handler = CommandHandler(mother_pipe)

    recognition = CommandRecognition(child_pipe)
    recognition.start()

    handler.dummy_loop()

    print('Stopped dummy loop')
    recognition.stop_process()


if __name__ == '__main__':
    main()
