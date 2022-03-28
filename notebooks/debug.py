
import pandas as pd
import numpy as np
#sktime
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate


from sklearn.linear_model import RidgeClassifierCV
a = pd.Series(np.arange(1000000))
b = pd.Series(np.arange(3883))
c = pd.Series(np.arange(999293))


X = pd.DataFrame({'x': [a,b,c], 'y': [2*a,3*b,4*c]})
y = np.array([1,2,3])

rkt = MiniRocketMultivariate(n_jobs=-1)
rkt.fit(X)
X_t = rkt.transform(X)

nans = np.argwhere(np.isnan(X_t.values))
nans = np.unique(nans[:,1])
X_t = np.delete(arr=X_t.values, obj=nans, axis=1)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
classifier.fit(X_t, y)

from sktime import show_versions; show_versions()
