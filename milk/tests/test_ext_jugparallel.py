import jug
from jug import value
import jug.options
from jug.tests.task_reset import task_reset

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
    assert len(alltasks) == 5

    jug.jug.execute(store, options)
    assert len(value(space['clustered'])) == 2
