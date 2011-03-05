try:
    import milksets
    del milksets
except ImportError:
    import sys
    sys.stderr.write('''\
    Could not import milksets.

    This companion package does not provide any functionality, but
    is necessary for some of the testing.''')


def run():
    import nose
    nose.main()

