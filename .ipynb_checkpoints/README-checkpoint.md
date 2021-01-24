# Predicting Credit Risk using Machine Learning

This Machine-Learning example uses a variety of credit-related risk factors to predict a potential client's credit risk.  Machine Learning models include Logistic Regression, Balanced Random Forest and EasyEnsemble, and a variety of re-sampling techniques are used (Oversampling/SMOTE, Undersampling/Cluster Centroids, and SMOTEENN) to re-sample the data.  Evaluation metrics like the accuracy score, classification report and confusion matrix are generated to compare models and determine which suits this particular set of data best.
![credit](https://www.badcredit.org/wp-content/uploads/2019/05/cash-loans-for-no-credit-feat.jpg?1)

---

## Data Pre-Processing:
After reading in the original dataset and converting a few columns, we were left the following dataset.  Here are just a few of the features, along with the target column "loan_status":

![dataframe](/Screenshots/dataframe.png?raw=true)

The data was then split into X and y datframes, training/testing sets, and then scaled.  We were now ready to try different re-sampling techniques.

---

## Re-Sampling Techniques:

### Naive Random Oversampling:

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=1)
    X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
    
Next use a Logistic Regression model to train the dataset and make predictions:

    from sklearn.linear_model import LogisticRegression
    over_model = LogisticRegression(solver='lbfgs', random_state=1)
    over_model.fit(X_resampled, y_resampled)
    predictions = over_model.predict(X_test_scaled)
    
Calculating Balanced Accuracy Score:

    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy_score(y_test, predictions)
    
    # Output = 0.6595245577351527
    
Generating Classification Report:

    from imblearn.metrics import classification_report_imbalanced
    print(classification_report_imbalanced(y_test, predictions))
    
The above code results in the following classification report:
![class_report](/Screenshots/class_report.png?raw=true)

### SMOTE Oversampling:

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=1, sampling_strategy=1.0)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    
### Cluster Centroids Undersampling:

    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=1)
    X_resampled, y_resampled = cc.fit_resample(X_train_scaled, y_train)
    
### SMOTEENN Combination Over/Undersampling:

    from imblearn.combine import SMOTEENN
    sm = SMOTEENN(random_state=1)
    X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)
    
### Results:
Of the 4 models, the Logistic Regression model using the SMOTEENN Combination Re-Sampler had the best balanced accuracy score at 79.8%.  
Of the 4 models, the Logistic Regression model using the Smote Oversampler had the best recall score at 88%.  
Of the 4 models, both the Logistic Regression model using the SMOTEENN Combination Re-Sampler and the model using the Smote Oversampler had the best geometric mean scores at 79%.

---

## Re-Sampling Using Ensemble Learners:

### Balanced Random Forest Classifier:

    from imblearn.ensemble import BalancedRandomForestClassifier
    brf_model = BalancedRandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=2, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1, oob_score=False, random_state=1, replacement=False, sampling_strategy='auto', verbose=0, warm_start=False)
    brf_model.fit(X_train, y_train)
    predictions = brf_model.predict(X_test_scaled)
    
### Feature Importance Table:

    importances = pd.DataFrame(brf_model.feature_importances_, index = X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False)
    
The above code results in the following Feature Importance Table (this shows top 5 features):
![importance](/Screenshots/importance.png?raw=true)

### Easy Ensemble Classifier:

    from imblearn.ensemble import EasyEnsembleClassifier
    ee_model = EasyEnsembleClassifier(base_estimator=None, n_estimators=100, n_jobs=1, random_state=1, replacement=False, sampling_strategy='auto', verbose=0, warm_start=False)
    ee_model.fit(X_train, y_train)
    predictions = ee_model.predict(X_test_scaled)
    
### Results:
Of the 2 models, the Easy Ensemble classifier had the best balanced accuracy score at 93%.  
Of the 2 models, the Easy Ensemble classifier had the best recall score at 94%.  
Of the 2 models, the Easy Ensemble classifier had the best geometric mean score at 93%.  
The top three features are total_rec_prncp (8.3%), total_pymnt_inv (6.9%) and last_pymnt_amnt (6.1%).  