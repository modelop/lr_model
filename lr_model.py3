# fastscore.schema.0: close_price
# fastscore.schema.1: tagged_double


import numpy as np
import pickle
from sklearn.linear_model import LinearRegression


# modelop.init
def begin():
    global lr
    global window, window_size
    window = []
    window_size = 15
    with open('lr_pickle1.pkl', 'rb') as f:
        lr = pickle.load(f)

# modelop.score
def action(x):
    global window, window_size
    x = x['Close']
    window = window[1-window_size:] + [x]
    if len(window) < window_size:
        yield {"name": "price", "value":x}
    else:
        X = np.array([window])
        y = lr.predict(X)
        yield {"name":"price", "value": y[0,0]}
