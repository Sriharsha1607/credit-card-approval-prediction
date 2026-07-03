from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Decision Tree Function
def d_tree(xtrain, xtest, ytrain, ytest):
    dt = DecisionTreeClassifier(random_state=42)

    dt.fit(xtrain, ytrain)

    ypred = dt.predict(xtest)

    print("***** Decision Tree Classifier *****")
    print("Accuracy :", accuracy_score(ytest, ypred))

    print("\nConfusion Matrix")
    print(confusion_matrix(ytest, ypred))

    print("\nClassification Report")
    print(classification_report(ytest, ypred))


# Random Forest Function
def random_forest(xtrain, xtest, ytrain, ytest):
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(xtrain, ytrain)

    ypred = rf.predict(xtest)

    print("***** Random Forest Classifier *****")
    print("Accuracy :", accuracy_score(ytest, ypred))

    print("\nConfusion Matrix")
    print(confusion_matrix(ytest, ypred))

    print("\nClassification Report")
    print(classification_report(ytest, ypred))


# Call the functions
d_tree(X_train, X_test, y_train, y_test)
random_forest(X_train, X_test, y_train, y_test)