data
----
Flexibly fold and pad input data into sequences of equal length and then
wrap them into custom data producers to be used in the training loop as well
as for validation.

Refer to the pertinent `documentation <https://yedivanseven.github.io/swak/pt/train.html>`_
of the `swak <https://github.com/yedivanseven/swak>`_ package (available on
`PyPI <https://pypi.org/project/swak/>`_) for how to best use these data
producers in the best training loop.


.. autoclass:: slangmod.ml.TestSequenceFolder
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: slangmod.ml.TestData
   :members:
   :inherited-members:
   :show-inheritance:


.. autoclass:: slangmod.ml.TrainSequenceFolder
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: slangmod.ml.TrainData
   :members:
   :inherited-members:
   :special-members: __call__
   :show-inheritance:
