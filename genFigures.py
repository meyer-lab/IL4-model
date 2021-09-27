#!/usr/bin/env python3

import sys
import logging
import time
import matplotlib
matplotlib.use('AGG')

fdir = './output/'

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if __name__ == '__main__':
    start = time.time()
    nameOut = 'figure' + sys.argv[1]

    exec('from src.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    logging.info('%s is done after %s seconds.', nameOut, time.time() - start)
