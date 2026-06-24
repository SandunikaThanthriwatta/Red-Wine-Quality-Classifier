from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def train_random_forest(x_train, y_train, n_estimators=100):
    rf_regressor = RandomForestClassifier(n_estimators=n_estimators)
    rf_regressor.fit(x_train, y_train)
    return rf_regressor


def train_svc(x_train, y_train):
    svc_regressor = SVC()
    svc_regressor.fit(x_train, y_train)
    return svc_regressor


def train_svc_tuned(x_train, y_train, C=1.2, gamma=0.9, kernel='rbf'):
    """SVC with best parameters found via GridSearchCV."""
    svc_regressor2 = SVC(C=C, gamma=gamma, kernel=kernel)
    svc_regressor2.fit(x_train, y_train)
    return svc_regressor2
