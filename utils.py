# Creating the utility function to separate categorical and numerical features
# This utilty function also checks the cardinality of the categorical features

def check_categorical(X):
    categorical_cols = [cname for cname in X.columns if
                        X[cname].nunique() < 100 and 
                        X[cname].dtype == "object"]
    return categorical_cols


def check_numerical(X):
    numerical_cols = [cname for cname in X.columns if 
                    X[cname].dtype in ['int64', 'float64']]
    return numerical_cols
