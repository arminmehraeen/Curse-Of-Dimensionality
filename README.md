# Curse-Of-Dimensionality
Curse of dimensionality ( python ) 

# Code

    import math
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt

    dims = np.arange(2,50,2)

    result = []
    number = 100

    for i in dims:

        y = []
        x = np.random.rand(number,i)

        for m in range(number):
            for n in range(number):
                if m != n:
                    y.append(math.dist(x[m],x[n]))
        y.sort()
        y = list(y for y,_ in itertools.groupby(y))

        result.append(math.log((( max(y) - min(y)) / min(y) ), 10))

    plt.plot(dims,result)
    plt.ylabel('Log(max-min/min)')
    plt.xlabel('Dims')
    plt.show()
