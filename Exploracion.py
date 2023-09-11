import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings("ignore", category=FutureWarning)

# Leer los datos
df = pd.read_csv("star_classification.csv")
df_describe = df.describe().transpose()
#df_describe.to_csv('describe2.csv', index=False)


X = df.drop('class', axis=1)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X.values))
X_describe = X.describe().transpose()
X_describe.to_csv('Xdescribe2.csv', index=False)
