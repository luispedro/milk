===================================
Feature Normalisation and Selection
===================================

For many problems, feature normalisation and selection is a 

Simple Normalisations
---------------------

Fill in ``NaNs`` and ``Infs``: the ``checkfinite()`` learner does this. This
learner does not use any of its input features: it always returns the same
model.

Whiten
------

Checkout the functions ``zscore()`` if you have a feature matrix or the
``zscore_normalise()`` learner.

Stepwise Discriminant Analysis
------------------------------

Stepwise Discriminant Analysis (SDA) is a simple feature selection method. It
is supervised and independent of the downstream classifier.

**Important Note**: SDA does not work well if your features are linearly
dependent. Filter out linearly dependent features before calling SDA (use
``linearly_dependent_features``).

