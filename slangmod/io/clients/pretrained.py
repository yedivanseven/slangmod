import readline  # noqa: F401
from typing import TypedDict
from collections.abc import Callable
from swak.misc import ArgRepr
from .styles import Style, style
from ...config import config

__all__ = [
    'PreTrainedClient',
    'pre_trained_client'
]

type Generator = Callable[[str], tuple[str, bool]]
class RoleText(TypedDict):
    role: str
    text: str
type Conversation = list[RoleText]


class PreTrainedClient(ArgRepr):
    """Chat with a pre-trained model.

    Parameters
    ----------
    style: Style, optional.
        A fully configured ``Style`` instance, i.e., a particular style to
        format user input in. Defaults to stripping whitespace characters and
        appending a space.
    system: str, optional
        The system prompt to prepend to the conversation.
        Default top an emtpy string.
    user: str, optional
        What the user should be called. Defaults to "USR".
    bot: str, optional
        What the model should be called. Defaults to "BOT".
    stop: str, optional
        A particular user intput that will stop the conversation and terminate
        the program. Defaults to "Stop!".
    eos_string: str, optional
        String that will be attached to model answers if it indicates that it
        has reached an end-of-sequence. Defaults to two consecutive new-line
        characters.

    Important
    ---------
    Make sure that the `eos_string` is consistent with how you prepared your
    corpus and how you set up your tokenizer. It must be tokenized as the
    end-of-sequence token.

    """

    def __init__(
            self,
            style: Style = Style(),
            system: str ='',
            user: str = 'USR',
            bot: str = 'BOT',
            stop: str = 'Stop!',
            eos_string: str = '\n\n'
    ) -> None:
        super().__init__(style, system[:10], user, bot, stop, eos_string)
        self.style = style
        self.system = system.lstrip()
        self.user = user.strip().upper()
        self.bot = bot.strip().upper()
        self.stop = stop.strip()
        self.eos_string = eos_string
        self.conversation = [(self.bot, self.system)] if self.system else []

    @property
    def flat(self):
        """The conversation flattened into a single string fed to the model."""
        return ''.join([text for _, text in self.conversation])

    @property
    def _pad(self) -> int:
        return max(len(self.user), len(self.bot))

    def __call__(self, generate: Generator) -> Conversation:
        """Chat with a pretrained model in the given format.

        Parameters
        ----------
        generate: Generator
            A wrapper around your model that accepts a string (the conversation
            so far) and returns a tuple with the model answer as well as a
            boolean indicating whether the model answer terminates with an
            end-of-sequence token or whether the model has more to say.

        Returns
        -------
        list
            The chat history as a list of dicts with their "role" keys having
            values `user` or `bot` and their "text" values containing the
            respective text.

        """
        if self.system:
            print(self.system)  # noqa: T201

        terminates = True

        while True:
            prompt = input(f'[{self.user:>{self._pad}}]> ')

            if prompt.strip() == '' and terminates:
                continue
            elif prompt.strip() == self.stop:
                break

            self.conversation.append((self.user, self.style(prompt)))
            generated, terminates = generate(self.flat)

            answer = generated + (self.eos_string if terminates else '')
            self.conversation.append((self.bot, answer))

            suffix = '' if terminates else ' ...\n'
            reply = answer + suffix

            print(f'\n[{self.bot:>{self._pad}}]>', reply, end='')  # noqa: T201
        return [{'role': role, 'text': txt} for role, txt in self.conversation]


# Provide a ready-to-use instance of the PreTrainedClient
pre_trained_client = PreTrainedClient(
    style=style,
    system=config.chat.system,
    user=config.chat.user,
    bot=config.chat.bot,
    stop=config.chat.stop,
    eos_string=config.tokens.eos_string
)
