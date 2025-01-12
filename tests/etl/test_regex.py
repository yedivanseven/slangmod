import pickle
import unittest
import re
from slangmod.etl import RegexReplacer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.pattern = r'a{2}'
        self.repl = '[UNK]'
        self.replace = RegexReplacer(self.pattern, self.repl)

    def test_has_pattern(self):
        self.assertTrue(hasattr(self.replace, 'pattern'))

    def test_pattern(self):
        self.assertIsInstance(self.replace.pattern, re.Pattern)
        self.assertEqual(self.pattern, self.replace.pattern.pattern)

    def test_has_repl(self):
        self.assertTrue(hasattr(self.replace, 'repl'))

    def test_repl(self):
        self.assertEqual(self.repl, self.replace.repl)

    def test_has_flags(self):
        self.assertTrue(hasattr(self.replace, 'flags'))

    def test_flags(self):
        self.assertIsInstance(self.replace.flags, int)
        self.assertEqual(0, self.replace.flags)

    def test_has_count(self):
        self.assertTrue(hasattr(self.replace, 'count'))

    def test_count(self):
        self.assertIsInstance(self.replace.count, int)
        self.assertEqual(0, self.replace.count)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        pattern = r'a{2}'
        repl = '[UNK]'
        self.flags = 2
        self.count = 5
        self.replace = RegexReplacer(pattern, repl, self.flags, self.count)

    def test_flags(self):
        self.assertEqual(self.flags, self.replace.flags)

    def test_count(self):
        self.assertEqual(self.count, self.replace.count)


class TestUsage(unittest.TestCase):

    def test_replace_all_matches(self):
        replace = RegexReplacer('a{2}', '[UNK]')
        text = 'Hellaao WoraaldAA!'
        actual = replace(text)
        expected = 'Hell[UNK]o Wor[UNK]ldAA!'
        self.assertEqual(expected, actual)

    def test_replace_count_matches(self):
        replace = RegexReplacer('a{2}', '[UNK]', count=1)
        text = 'Hellaao WoraaldAA!'
        actual = replace(text)
        expected = 'Hell[UNK]o WoraaldAA!'
        self.assertEqual(expected, actual)

    def test_replace_flag_matches(self):
        replace = RegexReplacer('a{2}', '[UNK]', flags=re.IGNORECASE)
        text = 'Hellaao WoraaldAA!'
        actual = replace(text)
        expected = 'Hell[UNK]o Wor[UNK]ld[UNK]!'
        self.assertEqual(expected, actual)

    def test_replace_function_matches(self):

        def repl(matches):
            return f'[{type(matches).__name__}]'

        replace = RegexReplacer('a{2}', repl)
        text = 'Hellaao WoraaldAA!'
        actual = replace(text)
        expected = 'Hell[Match]o Wor[Match]ldAA!'
        self.assertEqual(expected, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        replace = RegexReplacer(r'a{2}', '[UNK]')
        expected = "RegexReplacer('a{2}', '[UNK]', 0, 0)"
        self.assertEqual(expected, repr(replace))

    def test_custom_repr(self):
        replace = RegexReplacer(r'a{2}', '[UNK]', 2, 5)
        expected = "RegexReplacer('a{2}', '[UNK]', 2, 5)"
        self.assertEqual(expected, repr(replace))

    def test_pickle_works(self):
        replace = RegexReplacer(r'a{2}', '[UNK]', 2, 5)
        _ = pickle.loads(pickle.dumps(replace))


if __name__ == '__main__':
    unittest.main()
