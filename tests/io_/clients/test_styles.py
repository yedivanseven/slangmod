import pickle
import unittest
from slangmod.io.clients.styles import Style, space, quote, paragraph, dialogue
from slangmod.config import config


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.style = Style()

    def test_has_template(self):
        self.assertTrue(hasattr(self.style, 'template'))

    def test_template(self):
        self.assertEqual('{} ', self.style.template)

    def test_has_strip(self):
        self.assertTrue(hasattr(self.style, 'strip'))

    def test_strip(self):
        self.assertIsNone(self.style.strip)


class TestAttributes(unittest.TestCase):

    def test_template(self):
        obj = object()
        style = Style(obj)
        self.assertIs(style.template, obj)

    def test_strip(self):
        obj = object()
        style = Style(strip=obj)
        self.assertIs(style.strip, obj)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.style = Style()

    def test_callable(self):
        self.assertTrue(callable(self.style))

    def test_prompt(self):
        expected = 'Hello world!'
        actual = self.style('  ' + expected + '   ')
        self.assertEqual(expected + ' ', actual)

    def test_prompt_stripped(self):
        expected = 'Hello world!'
        actual = self.style(expected)
        self.assertEqual(expected + ' ', actual)


class TestCustomAttributes(unittest.TestCase):

    def test_template(self):
        style = Style('  !{}# ')
        prompt = 'Hello world!'
        expected = '  !Hello world!# '
        actual = style(prompt)
        self.assertEqual(expected, actual)

    def test_strip(self):
        style = Style('  "{}!" ', '#?')
        prompt = '      #Hello world?#     '
        expected = '  "Hello world!" '
        actual = style(prompt)
        self.assertEqual(expected, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        style = Style()
        expected = "Style('{} ', None)"
        self.assertEqual(expected, repr(style))

    def test_custom_repr(self):
        style = Style('"{}" ', '"')
        expected = """Style('"{}" ', '"')"""
        self.assertEqual(expected, repr(style))

    def test_pickle_works(self):
        style = Style('"{}" ', '"')
        _ = pickle.loads(pickle.dumps(style))


class TestPresets(unittest.TestCase):

    def test_space_happy(self):
        actual = space('Hello World!')
        expected = 'Hello World! '
        self.assertEqual(expected, actual)

    def test_space_edge(self):
        actual = space('   Hello World!\n  \n')
        expected = 'Hello World! '
        self.assertEqual(expected, actual)

    def test_paragraph_happy(self):
        actual = paragraph('Hello World!')
        expected = f'Hello World! {config.tokens.eos_symbol}'
        self.assertEqual(expected, actual)

    def test_paragraph_edge(self):
        actual = paragraph('  \nHello World! \n\n ')
        expected = f'Hello World! {config.tokens.eos_symbol}'
        self.assertEqual(expected, actual)

    def test_quote_happy(self):
        actual = quote('Hello World!')
        expected = '"Hello World!," '
        self.assertEqual(expected, actual)

    def test_quote_edge(self):
        actual = quote('  "Hello World!"\n  \n')
        expected = '"Hello World!," '
        self.assertEqual(expected, actual)

    def test_dialogue_happy(self):
        actual = dialogue('Hello World!')
        expected = f'"Hello World!" {config.tokens.eos_symbol}'
        self.assertEqual(expected, actual)

    def test_dialogue_edge(self):
        actual = dialogue('\n"Hello World!"\n\n')
        expected = f'"Hello World!" {config.tokens.eos_symbol}'
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
