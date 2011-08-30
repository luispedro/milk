# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import multiprocessing

max_procs = 1
_used_procs = multiprocessing.Value('i', 1)
_plock = multiprocessing.Lock()

def set_max_processors(value=None):
    '''
    set_max_processors(value=None)

    Set the maximum number of processors to ``value`` (or to the number of
    physical CPUs if ``None``).

    Note that this is valid for the current process and its children, but not
    the parent.

    Parameters
    ----------
    value : int, optional
        Number of processors to use. Defaults to number of CPUs (as returned by
        ``multiprocessing.cpu_count()``).
    '''
    global max_procs
    if value is None:
        value = multiprocessing.cpu_count()
    max_procs = value

def get_proc():
    '''
    available = get_proc()

    Reserve a processor

    Returns
    -------
    available : bool
        True if a processor is available
    '''
    with _plock:
        if _used_procs.value >= max_procs:
            return False
        _used_procs.value += 1
        return True

def release_proc():
    '''
    release_proc()

    Returns a processor to the pool
    '''
    with _plock:
        _used_procs.value -= 1

def release_procs(n, count_current=True):
    '''
    release_procs(n, count_current=True)

    Returns ``n`` processors to the pool

    Parameters
    ----------
    n : int
        Number of processors to release
    count_current : bool, optional
        Whether the current processor is to be included in ``n`` (default: True)
    '''
    if count_current:
        n -= 1
    if n > 0:
        with _plock:
            _used_procs.value -= n

def get_procs(desired=None, use_current=True):
    '''
    n = get_procs(desired=None, use_current=True)

    Get the up to ``desired`` processors (use None for no maximum).

    Parameters
    ----------
    desired : int, optional
        Number of processors you wish. By default, there is no maximum
    use_current: bool, optional
        Whether to count the current processor, True by default.
    '''
    if desired is None:
        desired = 1024 # This should last a few years
    n = (1 if use_current else 0)
    while n < desired:
        if get_proc():
            n += 1
        else:
            return n
    return n

