from sklearn.linear_model import LogisticRegression 

# define baseline model 
def train_logistic_regression(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000, 
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model 

# LGR --> Fast , stable, interpretable, 
# easier to scale & explain than SVM