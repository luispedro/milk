try:
    import jug
    from jug import value
    import jug.options
    from jug.tests.task_reset import task_reset
except ImportError:
    from nose import SkipTest
    def task_reset(f):
        def g():
            raise SkipTest()
        return g

@task_reset
def test_nfoldcrossvalidation():
    store, space = jug.jug.init('milk/tests/data/jugparallel_jugfile.py', 'dict_store')
    options = jug.options.default_options
    jug.jug.execute(store, options)
    assert len(jug.value(space['classified'])) == 2
    assert len(jug.value(space['classified_wpred'])) ==3


@task_reset
def test_kmeans():
    from jug.task import alltasks
    store, space = jug.jug.init('milk/tests/data/jugparallel_kmeans_jugfile.py', 'dict_store')
    options = jug.options.default_options

    jug.jug.execute(store, options)
    assert len(value(space['clustered'])) == 2
