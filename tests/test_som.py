import numpy as np
import random
import random
import milk.unsupervised.som

def test_som():
    data = np.arange(100000, dtype=np.float32)
    grid = np.array([data.flat[np.random.randint(0, data.size)] for i in xrange(64*64)]).reshape((64,64,1))
    points = data[:100].copy().astype(np.float32).reshape((-1,1))
    grid2 = grid.copy()
    milk.unsupervised.som.putpoints(grid, points, L=0., R=1)
    assert np.all(grid == grid2)
    milk.unsupervised.som.putpoints(grid, points, L=.5, R=1)
    assert not np.all(grid == grid2)

