import jug
import jug.options

def test_ext_jugparallel():
    store, space = jug.jug.init('milk/tests/data/jugparallel_jugfile.py', 'dict_store')
    options = jug.options.default_options
    jug.jug.execute(store, options)
    assert len(jug.value(space['classified'])) == 2
    assert len(jug.value(space['classified_wpred'])) ==3

