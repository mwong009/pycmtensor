import os
import sys
import time

import numpy
from pytensor import config, function, shared
from pytensor import tensor as at

sys.path.insert(0, os.path.abspath(".."))
vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], at.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any(
    [
        isinstance(x.op, at.elemwise.Elemwise) and ("Gpu" not in type(x.op).__name__)
        for x in f.maker.fgraph.toposort()
    ]
):
    print("Used the cpu")
else:
    print("Used the gpu")
