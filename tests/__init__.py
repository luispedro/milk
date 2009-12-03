import glob
import os.path
try:
    import milksets
    del milksets
except ImportError:
    import sys
    sys.stderr.write('''\
    Could not import milksets.

    This companion package does not provide any functionality, but
    is necessary for some of the testing.''')
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
