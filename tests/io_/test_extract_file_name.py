import unittest
from slangmod.io import extract_file_name


class TestUsage(unittest.TestCase):

    def test_root(self):
        actual = extract_file_name('/test.txt')
        self.assertEqual('test.txt', actual)

    def test_one_level(self):
        actual = extract_file_name('/hello/test.txt')
        self.assertEqual('test.txt', actual)

    def test_two_levels(self):
        actual = extract_file_name('/hello/world/test.txt')
        self.assertEqual('test.txt', actual)


if __name__ == '__main__':
    unittest.main()
