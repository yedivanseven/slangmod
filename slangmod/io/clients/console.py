from typing import Self
from ...config import config
from ...ml.generators import Generator


class ConsoleClient:

    def __init__(self, system: str ='', stop: str = 'Stop!') -> None:
        self.stop = stop
        self.system = system.strip() + '\n\n' if system.strip() else ''
        self.history = [('bot', self.system)] if self.system else []

    @property
    def flat(self):
        return ''.join([msg[1] for msg in self.history])

    def __call__(self, generate: Generator) -> Self:
        print(self.system, end='')  # noqa: T201
        while True:
            prompt = input('User: ')
            if prompt == self.stop:
                break
            self.history.append(('user', generate.wrap(prompt)))
            answer = generate(self.flat)
            self.history.append(('bot', answer.strip() + '\n\n'))
            print('\nBot:', answer, '\n')  # noqa: T201
        return self


console_client = ConsoleClient(config.chat.system, config.chat.stop)
