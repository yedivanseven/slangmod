strategy
========
I *strongly* discourage trying to train a full language model on the first go.
Instead, I suggest starting small, that is, with (very) little data and with
a ridiculously small model. Experiment with the :doc:`training </cli/train/train>`
parameters until you have a good feeling for how to reach full convergence fast,
then slowly increase the size of the model and explore how these parameters scale.
Only then would I recommend increasing the amount data to train on, step by step.
