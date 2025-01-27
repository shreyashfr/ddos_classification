import pickle
import numpy as np
from imblearn.over_sampling import SMOTE

pickle_file_X = 'oversampled_data_SMOTE_X.pickle'
pickle_file_y = 'oversampled_data_SMOTE_y.pickle'
pickle_file = 'merged_data.pickle'

data_folder = 'D:/Research/IoT DDoS/'


if __name__ == "__main__":
    fp = open(data_folder + pickle_file, 'rb')
    features = pickle.load(fp)
    fp.close()
    print(features.shape)

    X = features[:, 0:-1]
    y = features[:, -1]
    print(X.shape, y.shape)

    print(np.isnan(X).sum())  # Check how many NaN values exist
    print(np.isinf(X).sum())  # Check how many infinity values exist

    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
    print(X.shape, y.shape)

    fp = open(data_folder + pickle_file_X, 'wb')
    pickle.dump(X, fp)
    fp.close()

    fp = open(data_folder + pickle_file_y, 'wb')
    pickle.dump(y, fp)
    fp.close()