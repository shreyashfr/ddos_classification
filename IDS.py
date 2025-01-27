import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA

pickle_file_X = 'oversampled_data_SMOTE_X.pickle'
pickle_file_y = 'oversampled_data_SMOTE_y.pickle'
data_folder = 'C:/Research/IoT DDoS/'


if __name__ == "__main__":
    fp = open(data_folder + pickle_file_X, 'rb')
    X = pickle.load(fp)
    fp.close()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    fp = open(data_folder + pickle_file_y, 'rb')
    y = pickle.load(fp)
    fp.close()

    print(X.shape, y.shape)
    print(np.isnan(X).sum())  # Check how many NaN values exist
    print(np.isinf(X).sum())  # Check how many infinity values exist

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    print(X_train.shape, X_test.shape)
    print(sum(y_train))

    #model = XGBClassifier(n_jobs=-1, n_estimators=100)
    #model = RandomForestClassifier()
    #model = SVC()
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, average='weighted')
    print('Random Forest', score, precision, precision_w, recall, recall_w, f1,f1_w)