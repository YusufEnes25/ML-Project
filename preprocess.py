import pandas as pd
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
import os

def main():
    train_data = pd.read_csv('X_train.csv')
    

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(train_data)
    train_poly = poly.transform(train_data)
    
    
    my_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    my_scaler.fit(train_poly)
    
   
    test_file = 'X_test.csv'
    if os.path.exists(test_file):
        test_data = pd.read_csv(test_file)
    else:
        print("no X_test.csv found, using X_train to test")
        test_data = pd.read_csv('X_train.csv')
        
    
    test_poly = poly.transform(test_data)
    scaled_stuff = my_scaler.transform(test_poly)
    
    
    out_df = pd.DataFrame(scaled_stuff)
    out_df.to_csv('X_test_preprocessed.csv', index=False)
    print("Preprocessing with QuantileTransformer done")

if __name__ == "__main__":
    main()