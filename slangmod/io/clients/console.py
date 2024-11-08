import readline  # noqa: F401
from swak.misc import ArgRepr
from ...config import config
from ...ml.generators import Generator


class ConsoleClient(ArgRepr):

    def __init__(
            self,
            system: str ='',
            user: str = 'user',
            bot: str = 'bot',
            stop: str = 'Stop!',
            eos_string: str = '\n\n'
    ) -> None:
        super().__init__('...', user, bot, stop, eos_string)
        self.system = system.lstrip()
        self.user = user.strip().capitalize()
        self.bot = bot.strip().capitalize()
        self.stop = stop
        self.eos_string = eos_string
        self.history = [(self.bot, self.system)] if self.system else []

    @property
    def flat(self):
        return ''.join([msg[1] for msg in self.history])

    def __call__(self, generate: Generator) -> list[str]:
        print(self.system)  # noqa: T201
        while True:
            prompt = input(f'{self.user}: ')
            if prompt == '':
                continue
            elif prompt == self.stop:
                break
            self.history.append((self.user, generate.style(prompt)))
            answer = generate(self.flat).rstrip() + self.eos_string
            self.history.append((self.bot, answer))
            print(f'\n{self.bot}:', answer, end='')  # noqa: T201
        return self.history


console_client = ConsoleClient(
    system=config.chat.system,
    user=config.chat.user,
    bot=config.chat.bot,
    stop=config.chat.stop,
    eos_string=config.tokens.eos_string
)
