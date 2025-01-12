import pickle
import unittest
from slangmod.io import CorpusFilter


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.filter = CorpusFilter('test')

    def test_has_part(self):
        self.assertTrue(hasattr(self.filter, 'part'))

    def test_part(self):
        self.assertEqual('test', self.filter.part)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.filter = CorpusFilter('test')

    def test_callable(self):
        self.assertTrue(callable(self.filter))

    def test_empty(self):
        actual = self.filter('')
        self.assertIsInstance(actual, bool)
        self.assertFalse(actual)

    def test_false(self):
        actual = self.filter('this_is-atrain_file.parquet')
        self.assertIsInstance(actual, bool)
        self.assertFalse(actual)

    def test_true(self):
        actual = self.filter('this_is-atest_file.parquet')
        self.assertIsInstance(actual, bool)
        self.assertTrue(actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.filter = CorpusFilter('test')

    def test_rerpr(self):
        expected = "CorpusFilter('test')"
        self.assertEqual(expected, repr(self.filter))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.filter))


if __name__ == '__main__':
    unittest.main()
