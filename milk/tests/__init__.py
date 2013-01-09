try:
    import milksets
    del milksets
except ImportError:
    import sys
    sys.stderr.write('''\
    Could not import milksets.

    This companion package does not provide any functionality, but
    is necessary for some of the testing.''')


def run(verbose=False):
    import nose
    from os import path
    currentdir = path.dirname(__file__)
    updir = path.join(currentdir, '..')
    argv = ['', '--exe', '-w', updir]
    if verbose:
        argv.append('--verbose')
    nose.run('milk', argv=argv)

