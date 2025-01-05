import readline  # noqa: F401
from swak.misc import ArgRepr
from ...config import config
from ...ml.generators import Generator


# ToDo: Rethink the logic with EOS here! Make sure the tokenizer get's it!
# ToDo: THe style should probably be part of the client, not the generator!
class ConsoleClient(ArgRepr):

    def __init__(
            self,
            system: str ='',
            user: str = 'USR',
            bot: str = 'BOT',
            stop: str = 'Stop!',
            eos_string: str = '\n\n'
    ) -> None:
        super().__init__(system[:10], user, bot, stop, eos_string)
        self.system = system.lstrip()
        self.user = user.strip().upper()
        self.bot = bot.strip().upper()
        self.stop = stop
        self.eos_string = eos_string
        self.history = [(self.bot, self.system)] if self.system else []

    @property
    def flat(self):
        return ''.join([msg[1] for msg in self.history])

    @property
    def space(self) -> int:
        return max(len(self.user), len(self.bot))

    def __call__(self, generate: Generator) -> list[str]:
        print(self.system)  # noqa: T201

        terminates = True

        while True:
            prompt = input(f'[{self.user:>{self.space}}]> ')

            if prompt == '' and terminates:
                continue
            elif prompt == self.stop:
                break

            # ToDO: Replace style here ...
            self.history.append((self.user, generate.style(prompt)))
            generated, terminates = generate(self.flat)

            if terminates:
                answer = generated.rstrip() + self.eos_string
            else:
                answer = generated
            self.history.append((self.bot, answer))

            suffix = '' if terminates else f'...{self.eos_string}'
            reply = answer + suffix

            print(f'\n[{self.bot:>{self.space}}]> ', reply, end='')  # noqa: T201
        return self.history


console_client = ConsoleClient(
    system=config.chat.system,
    user=config.chat.user,
    bot=config.chat.bot,
    stop=config.chat.stop,
    eos_string=config.tokens.eos_string
)
