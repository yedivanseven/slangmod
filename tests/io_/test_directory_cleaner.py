import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from slangmod.io import DirectoryCleaner


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.folder = 'directory'
        self.clean = DirectoryCleaner('directory')

    def test_has_folder(self):
        self.assertTrue(hasattr(self.clean, 'folder'))

    def test_folder(self):
        self.assertEqual(self.folder, self.clean.folder)

    def test_folder_stringified(self):
        clean = DirectoryCleaner(1)
        self.assertEqual('1', clean.folder)

    def test_folder_stripped(self):
        clean = DirectoryCleaner('  directory ')
        self.assertEqual('directory', clean.folder)


class TestUsageReturnPathFalse(unittest.TestCase):

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.subfolder = TemporaryDirectory(dir=self.folder.name)
        self.name = 'test.txt'
        self.file = self.folder.name + '/' + self.name
        with Path(self.file).open('wt') as file:
            file.write('Hello world!')
        with (Path(self.subfolder.name) / 'train.txt').open('wt') as file:
            file.write('Hello world!')

    def tearDown(self):
        self.subfolder.cleanup()
        self.folder.cleanup()
        del self.subfolder
        del self.folder

    def test_callable(self):
        clean = DirectoryCleaner(self.folder.name)
        self.assertTrue(callable(clean))

    def test_clean_nested(self):
        path = Path(self.folder.name)
        self.assertTrue(path.exists())
        self.assertTrue(list(path.iterdir()))

        clean = DirectoryCleaner(self.folder.name)
        actual = clean()

        self.assertTupleEqual((), actual)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))

    def test_clean_subdir(self):
        path = Path(self.subfolder.name)
        name = Path(self.subfolder.name).name
        self.assertTrue(path.exists())
        self.assertTrue(list(path.iterdir()))

        clean = DirectoryCleaner(self.folder.name)
        actual = clean(name)

        self.assertTupleEqual((), actual)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))

    def test_clean_file(self):
        path = Path(self.file)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_file())

        clean = DirectoryCleaner(self.file)
        actual = clean()

        self.assertTupleEqual((), actual)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))

    def test_clean_sub_file(self):
        path = Path(self.file)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_file())

        clean = DirectoryCleaner(self.folder.name)
        actual = clean(self.name)

        self.assertTupleEqual((), actual)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))

    def test_create_new(self):
        clean = DirectoryCleaner(self.folder.name)
        actual = clean('new')

        self.assertTupleEqual((), actual)

        path = Path(self.folder.name) / 'new'
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))


class TestUsageReturnPathTrue(unittest.TestCase):

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.subfolder = TemporaryDirectory(dir=self.folder.name)
        self.name = 'test.txt'
        self.file = self.folder.name + '/' + self.name
        with Path(self.file).open('wt') as file:
            file.write('Hello world!')
        with (Path(self.subfolder.name) / 'train.txt').open('wt') as file:
            file.write('Hello world!')

    def tearDown(self):
        self.subfolder.cleanup()
        self.folder.cleanup()
        del self.subfolder
        del self.folder

    def test_callable(self):
        clean = DirectoryCleaner(self.folder.name, True)
        self.assertTrue(callable(clean))

    def test_clean_nested(self):
        path = Path(self.folder.name)
        self.assertTrue(path.exists())
        self.assertTrue(list(path.iterdir()))

        clean = DirectoryCleaner(self.folder.name, True)
        actual = clean()

        self.assertEqual(self.folder.name, actual)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))

    def test_clean_subdir(self):
        path = Path(self.subfolder.name)
        name = Path(self.subfolder.name).name
        self.assertTrue(path.exists())
        self.assertTrue(list(path.iterdir()))

        clean = DirectoryCleaner(self.folder.name, True)
        actual = clean(name)

        self.assertEqual(self.subfolder.name, actual)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))

    def test_clean_file(self):
        path = Path(self.file)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_file())

        clean = DirectoryCleaner(self.file, True)
        actual = clean()

        self.assertEqual(self.file, actual)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))

    def test_clean_sub_file(self):
        path = Path(self.file)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_file())

        clean = DirectoryCleaner(self.folder.name, True)
        actual = clean(self.name)

        self.assertEqual(self.file, actual)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))

    def test_create_new(self):
        clean = DirectoryCleaner(self.folder.name, True)
        actual = clean('new')

        expected = self.folder.name + '/new'
        self.assertEqual(expected, actual)

        path = Path(expected)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.load = DirectoryCleaner('directory')

    def test_repr(self):
        expected = "DirectoryCleaner('directory')"
        self.assertEqual(expected, repr(self.load))

    def tet_pickle_works(self):
        _ = pickle.dumps(self.load)


if __name__ == '__main__':
    unittest.main()
