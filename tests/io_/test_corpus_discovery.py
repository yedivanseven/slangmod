import pickle
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from slangmod.io import CorpusDiscovery


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.discover = CorpusDiscovery()

    def test_has_folder(self):
        self.assertTrue(hasattr(self.discover, 'folder'))

    def test_folder(self):
        self.assertEqual('', self.discover.folder)

    def test_has_types(self):
        self.assertTrue(hasattr(self.discover, 'types'))

    def test_types(self):
        self.assertTupleEqual(('',), self.discover.types)

    def test_has_suffix(self):
        self.assertTrue(hasattr(self.discover, 'suffix'))

    def test_suffix(self):
        self.assertEqual('parquet', self.discover.suffix)

    def test_has_not_found(self):
        self.assertTrue(hasattr(self.discover, 'not_found'))

    def test_not_found(self):
        self.assertEqual('raise', self.discover.not_found)


class TestCustomAttributes(unittest.TestCase):

    def test_folder(self):
        discover = CorpusDiscovery('folder')
        self.assertEqual('folder', discover.folder)

    def test_folder_stringified(self):
        discover = CorpusDiscovery(42)
        self.assertEqual('42', discover.folder)

    def test_folder_stripped(self):
        discover = CorpusDiscovery('  folder ')
        self.assertEqual('folder', discover.folder)

    def test_types(self):
        discover = CorpusDiscovery('', 'a', 'b', 'c')
        self.assertTupleEqual(('a', 'b', 'c'), discover.types)

    def test_types_stringified(self):
        discover = CorpusDiscovery('', 1, 2, 3)
        self.assertTupleEqual(('1', '2', '3'), discover.types)

    def test_assert_types_stripped(self):
        discover = CorpusDiscovery('', ' a', 'b ', '  c ')
        self.assertTupleEqual(('a', 'b', 'c'), discover.types)

    def test_suffix(self):
        discover = CorpusDiscovery(suffix='suffix')
        self.assertEqual('suffix', discover.suffix)

    def test_suffix_stringified(self):
        discover = CorpusDiscovery(suffix=42)
        self.assertEqual('42', discover.suffix)

    def test_suffix_stripped(self):
        discover = CorpusDiscovery(suffix=' .suffix .  ')
        self.assertEqual('suffix', discover.suffix)

    def test_not_found(self):
        discover = CorpusDiscovery(not_found='not_found')
        self.assertEqual('not_found', discover.not_found)

    def test_not_found_stringified(self):
        discover = CorpusDiscovery(not_found=42)
        self.assertEqual('42', discover.not_found)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.subfolder = TemporaryDirectory(dir=self.folder.name)
        self.suffix = 'txt'
        self.types = 'train', 'test'
        matching_1 = self.folder.name + '/test.txt'
        matching_2 = self.folder.name + '/train.txt'
        self.matches = {matching_1, matching_2}
        self.wrong_type = self.folder.name + '/validation.txt'
        self.wrong_suffix = self.folder.name + '/test.parquet'
        self.contents = self.matches | {self.wrong_type, self.wrong_suffix}
        for name in self.contents:
            with Path(name).open('wt') as file:
                file.write('Hello world!')

    def tearDown(self):
        self.subfolder.cleanup()
        self.folder.cleanup()
        del self.subfolder
        del self.folder

    def test_callable(self):
        discover = CorpusDiscovery()
        self.assertTrue(callable(discover))

    def test_return_type(self):
        discover = CorpusDiscovery(self.folder.name, suffix='*')
        actual = discover()
        self.assertIsInstance(actual, list)

    def test_sub_directories_filtered(self):
        discover = CorpusDiscovery(self.folder.name, suffix='*')
        actual = discover()
        self.assertSetEqual(self.contents, set(actual))

    def test_path_extended(self):
        discover = CorpusDiscovery('/', suffix='*')
        actual = discover(self.folder.name)
        self.assertSetEqual(self.contents, set(actual))

    def test_raises_on_directory_not_found(self):
        discover = CorpusDiscovery('non-existent')
        with self.assertRaises(FileNotFoundError):
            _ = discover()

    def test_raises_on_no_files_found(self):
        discover = CorpusDiscovery(self.folder.name, 'non-existent')
        with self.assertRaises(FileNotFoundError):
            _ = discover()

    def test_raises_on_suffix_not_found(self):
        discover = CorpusDiscovery(self.folder.name, suffix='non-existent')
        with self.assertRaises(FileNotFoundError):
            _ = discover()

    def test_warn_on_directory_not_found(self):
        discover = CorpusDiscovery('non-existent', not_found='warn')
        with self.assertWarns(UserWarning):
            actual = discover()
        self.assertListEqual([], actual)

    def test_warn_on_no_files_found(self):
        discover = CorpusDiscovery(
            self.folder.name,
            'non-existent',
            not_found='warn'
        )
        with self.assertWarns(UserWarning):
            actual = discover()
        self.assertListEqual([], actual)

    def test_warn_on_suffix_not_found(self):
        discover = CorpusDiscovery(
            self.folder.name,
            suffix='non-existent',
            not_found='warn'
        )
        with self.assertWarns(UserWarning):
            actual = discover()
        self.assertListEqual([], actual)

    def test_ignore_directory_not_found(self):
        discover = CorpusDiscovery('non-existent', not_found='ignore')
        actual = discover()
        self.assertListEqual([], actual)

    def test_ignore_no_files_found(self):
        discover = CorpusDiscovery(
            self.folder.name,
            'non-existent',
            not_found='ignore'
        )
        actual = discover()
        self.assertListEqual([], actual)

    def test_ignore_suffix_not_found(self):
        discover = CorpusDiscovery(
            self.folder.name,
            suffix='non-existent',
            not_found='ignore'
        )
        actual = discover()
        self.assertListEqual([], actual)

    def test_types_filtered(self):
        discover = CorpusDiscovery(self.folder.name, *self.types, suffix='*')
        actual = discover()
        self.assertSetEqual(self.matches | {self.wrong_suffix}, set(actual))

    def test_suffixes_filtered(self):
        discover = CorpusDiscovery(self.folder.name, suffix=self.suffix)
        actual = discover()
        self.assertSetEqual(self.matches | {self.wrong_type}, set(actual))

    def test_both_filtered(self):
        discover = CorpusDiscovery(
            self.folder.name,
            *self.types,
            suffix=self.suffix
        )
        actual = discover()
        self.assertSetEqual(self.matches, set(actual))


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        discover = CorpusDiscovery()
        expected = ("CorpusDiscovery('', '', suffix='parquet',"
                    " not_found='raise')")
        self.assertEqual(expected, repr(discover))

    def test_custom_repr(self):
        discover = CorpusDiscovery(
            'folder',
            'a', 'b',
            suffix='txt',
            not_found='warn'
        )
        expected = ("CorpusDiscovery('folder', 'a', 'b',"
                    " suffix='txt', not_found='warn')")
        self.assertEqual(expected, repr(discover))

    def test_pickle_works(self):
        discover = CorpusDiscovery(
            'folder',
            'a', 'b',
            suffix='txt',
            not_found='warn'
        )
        _ = pickle.loads(pickle.dumps(discover))


if __name__ == '__main__':
    unittest.main()
