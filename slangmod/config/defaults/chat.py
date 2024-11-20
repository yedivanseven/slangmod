from pathlib import Path
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Styles, Generators


def read_system_prompt(text: str) -> str:
    path = Path(text.strip())
    try:
        text_could_be_a_file = path.exists() and path.is_file()
    except OSError:
        text_could_be_a_file = False
    if text_could_be_a_file:
        with path.open() as file:
            text = file.read()
        return text
    return text


class Chat(JsonObject):
    style: Lower() = Styles.SPACE
    generator: Lower() = Generators.BEAM
    max_tokens: int = 256
    temperature: float = 1.0
    k: float = 0.1
    p: float = 0.8
    width: int = 16
    boost: float = 0.7
    user: str = 'user'
    bot: str = 'bot'
    stop: str = 'Stop!'
    system: read_system_prompt = '''
I had seen little of Holmes lately. My marriage had drifted us away
from each other. My own complete happiness, and the home-centred
interests which rise up around the man who first finds himself master
of his own establishment, were sufficient to absorb all my attention;
while Holmes, who loathed every form of society with his whole
Bohemian soul, remained in our lodgings in Baker Street, buried among
his old books, and alternating from week to week between cocaine and
ambition, the drowsiness of the drug, and the fierce energy of his
own keen nature. He was still, as ever, deeply attracted by the study
of crime, and occupied his immense faculties and extraordinary powers
of observation in following out those clues, and clearing up those
mysteries, which had been abandoned as hopeless by the official police.
From time to time I heard some vague account of his doings: of his
summons to Odessa in the case of the Trepoff murder, of his clearing
up of the singular tragedy of the Atkinson brothers at Trincomalee,
and finally of the mission which he had accomplished so delicately and
successfully for the reigning family of Holland. Beyond these signs of
his activity, however, which I merely shared with all the readers of
the daily press, I knew little of my former friend and companion.

'''
