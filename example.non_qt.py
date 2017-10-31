from CommandRecognition import CommandRecognition


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

        print('Stopped dummy loop')

    def print_alternatives(self, alternatives):
        print("Handler alternatives: ")
        for alternative in alternatives:
            print(alternative['transcript'], ': ', alternative['confidence'])


def main():
    # Init recognition service.
    recognition = CommandRecognition()
    handler_transport = recognition.get_external_transport()
    recognition.set_config_yaml('./recognition.config.yml')

    # Received commands handler.
    handler = CommandHandler(handler_transport)

    # Start recognizing.
    recognition.start()

    # Listen for transport to receive commands or transcription.
    handler.dummy_loop()

    # Shutdown recognition service.
    recognition.stop_process()


if __name__ == '__main__':
    main()
