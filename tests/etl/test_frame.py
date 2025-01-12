import pickle
import unittest
from unittest.mock import patch
from pandas import DataFrame, testing
from slangmod.etl import ToFrame


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.col = 'text'
        self.to_frame = ToFrame(self.col)

    def test_has_name(self):
        self.assertTrue(hasattr(self.to_frame, 'name'))

    def test_name(self):
        self.assertEqual(self.col, self.to_frame.name)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.to_frame, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.to_frame.kwargs)


class TestCustomAttributes(unittest.TestCase):

    def test_kwargs(self):
        expected = {'hello': 'world', 'answer': 42}
        to_frame = ToFrame('text', **expected)
        self.assertDictEqual(expected, to_frame.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.col = 'text'
        self.to_frame = ToFrame(self.col)

    def test_return_type(self):
        inp = [1, 2, 3]
        frame = self.to_frame(inp)
        self.assertIsInstance(frame, DataFrame)

    def test_return_shape(self):
        inp = [1, 2, 3]
        frame = self.to_frame(inp)
        self.assertTupleEqual((3, 1), frame.shape)

    def test_return_name(self):
        inp = [1, 2, 3]
        frame = self.to_frame(inp)
        self.assertEqual(self.col, frame.columns[0])

    def test_return_values(self):
        inp = [1, 2, 3]
        actual = self.to_frame(inp)
        expected = DataFrame(inp, columns=[self.col])
        testing.assert_frame_equal(expected, actual)

    @patch('slangmod.etl.frame.Series')
    def test_series_called_with_kwargs(self, mock):
        to_frame = ToFrame('text', answer=42)
        inp = [1, 2, 3]
        _ = to_frame(inp)
        mock.assert_called_once_with(inp, name='text', answer=42)

    def test_tuple(self):
        inp = [1, 2, 3]
        actual = self.to_frame(tuple(inp))
        expected = DataFrame(inp, columns=[self.col])
        testing.assert_frame_equal(expected, actual)

    def test_iterator(self):
        inp = [1, 2, 3]
        actual = self.to_frame(iter(inp))
        expected = DataFrame(inp, columns=[self.col])
        testing.assert_frame_equal(expected, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        to_frame = ToFrame('text')
        expected = "ToFrame('text')"
        self.assertEqual(expected, repr(to_frame))

    def test_custom_repr(self):
        to_frame = ToFrame('text', hello='world', answer=42)
        expected = "ToFrame('text', hello='world', answer=42)"
        self.assertEqual(expected, repr(to_frame))

    def test_pickle_works(self):
        to_frame = ToFrame('text', hello='world', answer=42)
        _ = pickle.loads(pickle.dumps(to_frame))


if __name__ == '__main__':
    unittest.main()
