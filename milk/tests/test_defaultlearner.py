import milk
def test_extra_arg():
    from milksets.wine import load
    features,labels = load()
    learner = milk.defaultlearner()
    model = learner.train(features[::2],labels[::2], extra_arg=5)
    assert model.apply(features[1]) < 12.
