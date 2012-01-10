try:
    import jug
    from jug import value
    import jug.options
    from jug.tests.utils import task_reset, simple_execute
except ImportError:
    from nose import SkipTest
    def task_reset(f):
        def g():
            raise SkipTest()
        return g

@task_reset
def test_nfoldcrossvalidation():
    store, space = jug.jug.init('milk/tests/data/jugparallel_jugfile.py', 'dict_store')
    simple_execute()
    assert len(jug.value(space['classified'])) == 2
    assert len(jug.value(space['classified_wpred'])) ==3


@task_reset
def test_kmeans():
    store, space = jug.jug.init('milk/tests/data/jugparallel_kmeans_jugfile.py', 'dict_store')
    simple_execute()
    assert len(value(space['clustered'])) == 2
