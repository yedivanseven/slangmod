import pickle
import unittest
from unittest.mock import patch
from slangmod.io.clients import PreTrainedClient, Style

ANSWER = 'Model answer.'


def generate(terminate: bool = True):

    def generator(_: str) -> tuple[str, bool]:
        return ANSWER, terminate

    return generator


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.client = PreTrainedClient()

    def test_has_style(self):
        self.assertTrue(hasattr(self.client, 'style'))

    def test_style(self):
        self.assertIsInstance(self.client.style, Style)

    def test_has_system(self):
        self.assertTrue(hasattr(self.client, 'system'))

    def test_system(self):
        self.assertEqual('', self.client.system)

    def test_has_user(self):
        self.assertTrue(hasattr(self.client, 'user'))

    def test_user(self):
        self.assertEqual('USR', self.client.user)

    def test_has_bot(self):
        self.assertTrue(hasattr(self.client, 'bot'))

    def test_bot(self):
        self.assertEqual('BOT', self.client.bot)

    def test_has_stop(self):
        self.assertTrue(hasattr(self.client, 'stop'))

    def test_stop(self):
        self.assertEqual('Stop!', self.client.stop)

    def test_has_eos_string(self):
        self.assertTrue(hasattr(self.client, 'eos_string'))

    def test_eos_string(self):
        self.assertEqual('\n\n', self.client.eos_string)

    def test_has_conversation(self):
        self.assertTrue(hasattr(self.client, 'conversation'))

    def test_conversation(self):
        self.assertListEqual([], self.client.conversation)

    def test_has_flat(self):
        self.assertTrue(hasattr(self.client, 'flat'))

    def test_flat(self):
        self.assertEqual('', self.client.flat)


class TestCustomAttributes(unittest.TestCase):

    def test_style(self):
        obj = object()
        client = PreTrainedClient(obj)
        self.assertIs(client.style, obj)

    def test_system(self):
        client = PreTrainedClient(system='Hello world!')
        self.assertEqual('Hello world!', client.system)

    def test_system_left_stripped(self):
        client = PreTrainedClient(system='  Hello world! ')
        self.assertEqual('Hello world! ', client.system)

    def test_user(self):
        client = PreTrainedClient(user='USER')
        self.assertEqual('USER', client.user)

    def test_user_stripped(self):
        client = PreTrainedClient(user='  USER ')
        self.assertEqual('USER', client.user)

    def test_user_upper(self):
        client = PreTrainedClient(user='user')
        self.assertEqual('USER', client.user)

    def test_bot(self):
        client = PreTrainedClient(bot='ASSISTANT')
        self.assertEqual('ASSISTANT', client.bot)

    def test_bot_stripped(self):
        client = PreTrainedClient(bot='  ASSISTANT ')
        self.assertEqual('ASSISTANT', client.bot)

    def test_bot_upper(self):
        client = PreTrainedClient(bot='assistant')
        self.assertEqual('ASSISTANT', client.bot)

    def test_stop(self):
        client = PreTrainedClient(stop='stop')
        self.assertEqual('stop', client.stop)

    def test_stop_stripped(self):
        client = PreTrainedClient(stop='  stop ')
        self.assertEqual('stop', client.stop)

    def test_eos_string(self):
        client = PreTrainedClient(eos_string='eos')
        self.assertEqual('eos', client.eos_string)

    def test_conversation(self):
        client = PreTrainedClient(system='Hello World!')
        self.assertListEqual(
            [(client.bot, 'Hello World!')],
            client.conversation
        )

    def test_flat(self):
        client = PreTrainedClient(system='Hello World!')
        self.assertEqual('Hello World!', client.flat)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.system = 'system '
        self.client = PreTrainedClient(Style('{} ', None), system=self.system)

    @patch('builtins.input', return_value='Stop!')
    @patch('builtins.print')
    def test_default_system_prompt_not_printed(self, out, _):
        client = PreTrainedClient()
        _ = client(generate(True))
        self.assertEqual(0, out.call_count)

    @patch('builtins.input', return_value='Stop!')
    @patch('builtins.print')
    def test_system_prompt_printed(self, out, _):
        _ = self.client(generate(True))
        self.assertEqual(1, out.call_count)
        self.assertEqual(self.system, out.call_args[0][0])

    @patch('builtins.input', return_value='Stop!')
    def test_user_input_prompt(self, inp):
        _ = self.client(generate(True))
        self.assertEqual(1, inp.call_count)
        expected = f'[{self.client.user}]> '
        self.assertEqual(expected, inp.call_args[0][0])

    @patch('builtins.input', return_value='Stop!')
    def test_user_input_prompt_extended(self, inp):
        client = PreTrainedClient(bot='assistant')
        _ = client(generate(True))
        self.assertEqual(1, inp.call_count)
        expected = f'[      {client.user}]> '
        self.assertEqual(expected, inp.call_args[0][0])

    @patch('builtins.input', return_value='  Stop! ')
    def test_user_input_stripped(self, _):
        _ = self.client(generate(True))

    @patch('builtins.input', return_value='Stop!')
    def test_conversation_not_extended_on_immediate_stop(self, _):
        _ = self.client(generate(True))
        self.assertEqual(self.system, self.client.flat)
        self.assertListEqual(
            [(self.client.bot, self.system)],
            self.client.conversation
        )

    @patch('builtins.input', side_effect=['', '', 'Stop!'])
    def test_reject_empty_user_input_when_terminates(self, inp):
        _ = self.client(generate(True))
        self.assertEqual(3, inp.call_count)
        self.assertEqual(self.system, self.client.flat)
        self.assertListEqual(
            [(self.client.bot, self.system)],
            self.client.conversation
        )

    @patch('builtins.input', side_effect=['Hello!', 'Stop!'])
    def test_conversation_appended_with_terminates(self, _):
        _ = self.client(generate(True))
        expected = [
            (self.client.bot, self.system),
            (self.client.user, 'Hello! '),
            (self.client.bot, ANSWER + self.client.eos_string)
        ]
        self.assertEqual(expected, self.client.conversation)
        expected = f'{self.system}Hello! {ANSWER}{self.client.eos_string}'
        self.assertEqual(expected, self.client.flat)

    @patch('builtins.input', side_effect=['Hello!', 'Stop!'])
    @patch('builtins.print')
    def test_output_answer_when_terminates(self, out, _):
        _ = self.client(generate(True))
        self.assertEqual(2, out.call_count)
        out.assert_called_with(
            f'\n[{self.client.bot}]>',
            f'{ANSWER}{self.client.eos_string}',
            end='')

    @patch('builtins.input', side_effect=['Hello!', 'Stop!'])
    @patch('builtins.print')
    def test_bot_prompt_extended(self, out, _):
        client = PreTrainedClient(system='system', user='user')
        _ = client(generate(True))
        self.assertEqual(2, out.call_count)
        out.assert_called_with(
            f'\n[ {self.client.bot}]>',
            f'{ANSWER}{self.client.eos_string}',
            end='')

    @patch('builtins.input', side_effect=['Hello!', 'Stop!'])
    def test_conversation_appended_with_not_terminates(self, _):
        _ = self.client(generate(False))
        expected = [
            (self.client.bot, self.system),
            (self.client.user, 'Hello! '),
            (self.client.bot, ANSWER)
        ]
        self.assertEqual(expected, self.client.conversation)
        expected = f'{self.system}Hello! {ANSWER}'
        self.assertEqual(expected, self.client.flat)

    @patch('builtins.input', side_effect=['Hello!', 'Stop!'])
    @patch('builtins.print')
    def test_output_answer_when_not_terminates(self, out, _):
        _ = self.client(generate(False))
        self.assertEqual(2, out.call_count)
        out.assert_called_with(
            f'\n[{self.client.bot}]>',
            f'{ANSWER} ...\n',
            end='')

    @patch('builtins.input', side_effect=['Hello!', 'Stop!'])
    def test_return_value(self, _):
        out = self.client(generate(True))
        hist = [
            (self.client.bot, self.system),
            (self.client.user, 'Hello! '),
            (self.client.bot, ANSWER + self.client.eos_string)
        ]
        expected = [{'role': role, 'text': text} for role, text in hist]
        self.assertListEqual(expected, out)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        client = PreTrainedClient()
        expected = ("PreTrainedClient(Style('{} ', None), '',"
                    " 'USR', 'BOT', 'Stop!', '\\n\\n')")
        self.assertEqual(expected, repr(client))

    def test_custom_repr(self):
        client = PreTrainedClient(
            Style("{}", '#'),
            'system',
            'user',
            'assistant',
            'stop',
            'eos'
        )
        expected = ("PreTrainedClient(Style('{}', '#'), 'system',"
                    " 'user', 'assistant', 'stop', 'eos')")
        self.assertEqual(expected, repr(client))

    def test_pickle_works(self):
        client = PreTrainedClient(
            Style("{}", '#'),
            'system',
            'user',
            'assistant',
            'stop',
            'eos'
        )
        _ = pickle.loads(pickle.dumps(client))


if __name__ == '__main__':
    unittest.main()
