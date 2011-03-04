import numpy as np
import random
from milk.unsupervised import som
from milk.unsupervised.som import putpoints, closest


def _slow_putpoints(grid, points, L=.2):
    for point in points:
        dpoint = grid-point
        y,x = np.unravel_index(np.abs(dpoint).argmin(), dpoint.shape)
        for dy in xrange(-4, +4):
            for dx in xrange(-4, +4):
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= grid.shape[0]:
                    continue
                if nx < 0 or nx >= grid.shape[1]:
                    continue
                L2 = L/(1+np.abs(dy)+np.abs(dx))
                grid[ny,nx] *= 1. - L2
                grid[ny,nx] += point*L2


def data_grid():
    np.random.seed(22)
    data = np.arange(100000, dtype=np.float32)
    grid = np.array([data.flat[np.random.randint(0, data.size)] for i in xrange(64*64)]).reshape((64,64,1))
    data = data.reshape((-1,1))
    return grid, data

def test_putpoints():
    grid, points = data_grid()
    points = points[:100]
    grid2 = grid.copy()
    putpoints(grid, points, L=0., R=1)
    assert np.all(grid == grid2)
    putpoints(grid, points, L=.5, R=1)
    assert not np.all(grid == grid2)

def test_against_slow():
    grid, points = data_grid()
    grid2 = grid.copy()
    putpoints(grid, points[:10], shuffle=False)
    _slow_putpoints(grid2.reshape((64,64)), points[:10])
    assert np.allclose(grid, grid2)


def test_som():
    N = 10000
    np.random.seed(2)
    data = np.array([np.arange(N), N/4.*np.random.randn(N)])
    data = data.transpose().copy()
    grid = som(data, (8,8), iterations=3, R=4)
    assert grid.shape == (8,8,2)
    y,x = closest(grid, data[0])
    assert 0 <= y < grid.shape[0]
    assert 0 <= x < grid.shape[1]

    grid2 = grid.copy()
    np.random.shuffle(grid2)
    full = np.abs(np.diff(grid2[:,:,0], axis=0)).mean()
    obs = np.abs(np.diff(grid[:,:,0], axis=0)).mean()
    obs2 = np.abs(np.diff(grid[:,:,0], axis=1)).mean()
    assert obs + 4*np.abs(obs-obs2) < full

