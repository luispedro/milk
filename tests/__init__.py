import glob
import os.path
basedir = os.path.dirname(__file__)
for mod in glob.glob(basedir + '/*.py'):
    mod = os.path.basename(mod)
    if mod == '__init__.py':
        continue
    mod = mod[:-len('.py')]
    _tmp = __import__(mod, globals(), locals(), [], level=-1)
    exec('%s = _tmp' % mod)
del glob
del os
del _tmp
