work in progress....


```python
import numpy as np
from sinkdiv import SinkDiv

# First measure.
a = np.random.rand(10)      # point masses
x = np.random.randn(10, 2)  # positions in 2d

# Second measure.
b = np.random.rand(10)      # point masses
y = np.random.randn(10)     # positions in 2d

# Specify sinkhorn divergence
divergence = SinkDivKL(eps=0.01, lam=1.0)

loss = divergence(a, x, b, y)
```

