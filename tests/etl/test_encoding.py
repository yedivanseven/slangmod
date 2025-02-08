import pickle
import unittest
from slangmod.etl import EncodingEnforcer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.encoding = 'ascii'
        self.encode = EncodingEnforcer(self.encoding)

    def test_has_encoding(self):
        self.assertTrue(hasattr(self.encode, 'encoding'))

    def test_encoding(self):
        self.assertEqual(self.encoding, self.encode.encoding)

    def test_has_repl(self):
        self.assertTrue(hasattr(self.encode, 'repl'))

    def test_repl(self):
        self.assertEqual(' ', self.encode.repl)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.encoding = 'ascii'
        self.repl = ' repl '
        self.encode = EncodingEnforcer(self.encoding, self.repl)

    def test_repl(self):
        self.assertEqual(self.repl, self.encode.repl)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.encode = EncodingEnforcer('ascii', '[UNK]')

    def test_callable(self):
        self.assertTrue(callable(self.encode))

    def test_empty_string(self):
        actual = self.encode('')
        self.assertEqual('', actual)

    def test_empty_multiline_string(self):
        actual = self.encode('''''')
        self.assertEqual('', actual)

    def test_nothing_to_replace(self):
        actual = self.encode('Hello World!')
        self.assertEqual('Hello World!', actual)

    def test_replace(self):
        text = 'Shikamaru Nara (奈良シカマル, Nara Shikamaru) is a fictional'
        expected = ('Shikamaru Nara ([UNK][UNK][UNK][UNK][UNK][UNK], '
                    'Nara Shikamaru) is a fictional')
        actual = self.encode(text)
        self.assertEqual(expected, actual)

    def test_newlines(self):
        text = 'Hello\nWorld!'
        actual = self.encode(text)
        self.assertEqual(text, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        encode = EncodingEnforcer('ascii')
        expected = "EncodingEnforcer('ascii', ' ')"
        self.assertEqual(expected, repr(encode))

    def test_custom_repr(self):
        encode = EncodingEnforcer('ascii', 'repl')
        expected = "EncodingEnforcer('ascii', 'repl')"
        self.assertEqual(expected, repr(encode))

    def test_pickle_works(self):
        encode = EncodingEnforcer('ascii', 'repl')
        _ = pickle.loads(pickle.dumps(encode))


if __name__ == '__main__':
    unittest.main()
