
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

data_train = pd.read_csv(r'training_data\train_split.csv')

classes = np.unique(data_train.iloc[:, -1])
print(classes)
weights = compute_class_weight(
    class_weight='balanced', classes=classes, y=data_train.iloc[:, -1])
print(weights)
class_weights = dict(zip(classes, weights))
print(class_weights)
