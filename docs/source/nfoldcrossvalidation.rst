================
Cross-validation
================

Cross validation is one of the better ways to evaluate the performance of
supervised classification.

Cross validation consists of separating the data into fold (hence the name
_n_-fold cross-validation, where _n_ is a positive integer). For the purpose o
this discussion, we consider 10 folds. In the first round, we leave the first
fold out. This means we train on the other 9 folds and then evaluate the model
on this left-out fold. On the second round, we leave the second fold out. This
continues until every fold has been left out exactly once.

Milk support what is often explicitly called *stratified cross validation*,
which means that it takes the class distributions into account (so that, in 10
fold cross validation, each fold will have 10% of each class per round).

An additional functionality, not normally found in machine learning packages or
in machine learning theory, but very useful in practice is the use of the
``origins`` parameter. Every datapoint can have an associated *origin*. This is
a an integer and its meaning is the following: all examples with the same
origin will be in the same fold (so testing will never be performed where there
was an object of the same origin used for training).

This can model cases such as the following: you have collected patient data,
which includes both some health measurement and an outcome of interest (for
example, how the patient was doing a year after the initial exam). You wish to
evaluate a supervised classification algorithm for predicting outcomes. In
particular, you wish for an estimate of how well the system would perform on
patients in any location (you know that the data collection has some site
effects, perhaps because each person runs the test a little bit differently).
Fortunately, you have the data to test this: the patients come from several
clinics. Now, you set each patient origin to be the ID of the clinic and
evaluate the per patient accuracy.


API Documentation
-----------------

.. automodule:: milk.measures.nfoldcrossvalidation
    :members:

