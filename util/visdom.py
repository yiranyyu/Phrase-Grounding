try:
    import visdom
except ImportError:
    raise RuntimeError("No visdom package is found. Please install it with command: \n pip install visdom")
import numpy as np


def vis_init(env='main', clear=True):
    vis = visdom.Visdom(env=env, port=4399)
    clear and vis.close()
    return vis


def vis_create(vis, title='', xlabel='', ylabel='', legend=None, npts=1):
    if npts == 1:
        return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
    else:
        return vis.line(X=np.array([npts * [1]]), Y=np.array([npts * [np.nan]]),
                        opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, legend=legend))
