import pickle
import unittest
from slangmod.io import FileTypeExtractor


class TestAttributes(unittest.TestCase):

    def test_one_file_type(self):
        extract = FileTypeExtractor('test')
        self.assertTrue(hasattr(extract, 'types'))
        self.assertTupleEqual(('test',), extract.types)

    def test_one_file_type_stringified(self):
        extract = FileTypeExtractor(1)
        self.assertTupleEqual(('1',), extract.types)

    def test_one_file_type_stripped(self):
        extract = FileTypeExtractor('  test ')
        self.assertTupleEqual(('test',), extract.types)

    def test_two_file_types(self):
        extract = FileTypeExtractor('test', 'train')
        self.assertTrue(hasattr(extract, 'types'))
        self.assertTupleEqual(('test', 'train'), extract.types)

    def test_two_file_types_stringified(self):
        extract = FileTypeExtractor(1, 2)
        self.assertTupleEqual(('1', '2'), extract.types)

    def test_two_file_types_stripped(self):
        extract = FileTypeExtractor('  test ', ' train  ')
        self.assertTupleEqual(('test', 'train'), extract.types)

    def test_three_file_types(self):
        extract = FileTypeExtractor('test', 'train', 'validation')
        self.assertTrue(hasattr(extract, 'types'))
        self.assertTupleEqual(('test', 'train', 'validation'), extract.types)

    def test_three_file_types_stringified(self):
        extract = FileTypeExtractor(1, 2, 3)
        self.assertTupleEqual(('1', '2', '3'), extract.types)

    def test_three_file_types_stripped(self):
        extract = FileTypeExtractor(' test  ', '  train ', ' validation ')
        self.assertTrue(hasattr(extract, 'types'))
        self.assertTupleEqual(('test', 'train', 'validation'), extract.types)


class TestUsage(unittest.TestCase):

    def test_one_file_type_no_match(self):
        extract = FileTypeExtractor('test')
        with self.assertRaises(ValueError):
            _ = extract('hello world.test')

    def test_one_file_type_one_match(self):
        extract = FileTypeExtractor('test')
        actual = extract('hello test.world')
        self.assertEqual('test', actual)

    def test_two_file_types_no_match(self):
        extract = FileTypeExtractor('test', 'train')
        with self.assertRaises(ValueError):
            _ = extract('hello world.test')

    def test_two_file_types_one_match(self):
        extract = FileTypeExtractor('test', 'train')
        actual = extract('hello test.world')
        self.assertEqual('test', actual)

    def test_two_file_types_two_matches(self):
        extract = FileTypeExtractor('test', 'train')
        with self.assertRaises(ValueError):
            _ = extract('train test.world')

    def test_three_file_types_no_match(self):
        extract = FileTypeExtractor('test', 'train', 'validation')
        with self.assertRaises(ValueError):
            _ = extract('hello world.test')

    def test_three_file_types_one_match(self):
        extract = FileTypeExtractor('test', 'train', 'validation')
        actual = extract('hello test.world')
        self.assertEqual('test', actual)

    def test_three_file_types_two_matches(self):
        extract = FileTypeExtractor('test', 'train', 'validation')
        with self.assertRaises(ValueError):
            _ = extract('traintest.world')

    def test_three_file_types_three_matches(self):
        extract = FileTypeExtractor('test', 'train', 'validation')
        with self.assertRaises(ValueError):
            _ = extract('train test-validation.world')


class TestMisc(unittest.TestCase):

    def test_repr_one_file_type(self):
        extract = FileTypeExtractor('test')
        expected = "FileTypeExtractor('test')"
        self.assertEqual(expected, repr(extract))

    def test_repr_two_file_types(self):
        extract = FileTypeExtractor('test', 'train')
        expected = "FileTypeExtractor('test', 'train')"
        self.assertEqual(expected, repr(extract))

    def test_repr_three_file_types(self):
        extract = FileTypeExtractor('test', 'train', 'validation')
        expected = "FileTypeExtractor('test', 'train', 'validation')"
        self.assertEqual(expected, repr(extract))

    def test_pickle_works(self):
        extract = FileTypeExtractor('test', 'train', 'validation')
        _ = pickle.loads(pickle.dumps(extract))


if __name__ == '__main__':
    unittest.main()
