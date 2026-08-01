import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

def main():
    X_df = pd.read_csv('X_train.csv')
    y_df = pd.read_csv('y_train.csv')
    y = y_df.values.ravel()

    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_df)

    
    X_train, X_val, y_train, y_val = train_test_split(X_poly, y, test_size=0.2, random_state=42, stratify=y)

    
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scl = scaler.fit_transform(X_train)
    X_val_scl = scaler.transform(X_val)

    
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    
    param_grid = {
        'max_depth': [6, 7, 8],                  
        'learning_rate': [0.05, 0.08, 0.1],       
        'n_estimators': [300, 400, 500],          
        'subsample': [0.8],                   
        'colsample_bytree': [0.8],            
        'reg_lambda': [4.0, 5.0, 6.0]             
    }
    
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')
    
    print("Running the final Quantile tournament")
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1_macro', cv=6, verbose=1, n_jobs=-1)
    
    grid_search.fit(X_train_scl, y_train, sample_weight=weights)

    best_clf = grid_search.best_estimator_
    print("\nbest settings found:", grid_search.best_params_)

    preds = best_clf.predict(X_val_scl)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    
    print(f"\nFinal Validation Acc: {acc:.4f}")
    print(f"Final Validation F1:  {f1:.4f}")
    
    best_clf.save_model('model.ubj')
    print("Quantile XGBoost model saved to model.ubj")

if __name__ == "__main__":
    main()