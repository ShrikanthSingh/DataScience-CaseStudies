import argparse
import datetime
import lightgbm
import numpy as np
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


CATEGORICAL_FEATURES = [
    "hierarchy_level2_desc",
    "hierarchy_level3_desc", 
]
NUMERICAL_FEATURES = [
    "store_count",
    "total_cust_count",
    "unit_price_weekly",
    #"unit_discount_weekly"
]
TEXT_FEATURES = "art_name"

FLAG_FEATURES = [
    "low_stock_warning"
]

TARGET = "sold_qty_units"

model = Pipeline([
    ("feature_preprocessor", ColumnTransformer([
        
        ("one_hot", OneHotEncoder(), FLAG_FEATURES),

        ("categorical", OrdinalEncoder(handle_missing="return_nan"), CATEGORICAL_FEATURES),

        ("numerical", "passthrough", NUMERICAL_FEATURES),

        ("text", TfidfVectorizer(), TEXT_FEATURES),
        
    ])),
    
    ("regressor", lightgbm.LGBMRegressor(
        n_estimators=3000,
        objective="regression",
        num_leaves=50,
        max_depth=10,
        min_child_samples=60,
        learning_rate=0.09,
        colsample_bytree=0.6,
        verbosity=-1,
        extra_trees=True,
        metric="mape"
    )),
])

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def estimate_sample_weights(df, n):
    return ((df.date_of_day - df.date_of_day.min()).dt.days + 1) ** n

def parse_args():
    parser = argparse.ArgumentParser(description="Training production model")
    parser.add_argument(
        "--input-train-csv",
        required=True,
        help="Path to the CSV file contains training data. e.g. /home/data/train.csv",
    )
    parser.add_argument(
        "-om",
        "--output-model",
        required=True,
        help="Path to the file where trained model will be stored. e.g. /home/models/regressor.joblib",
    )
    parser.add_argument(
        "-ot",
        "--output-test-pred",
        required=True,
        help=(
            "Path to the file where predictions for the test data will be stored. "
            "e.g. /home/models/data/test_predictions.csv"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print('\n')
    print("--------Reading the data----------")
    data_sales = pd.read_csv(args.input_train_csv)
    data_sales['date_of_day'] = pd.to_datetime(data_sales['date_of_day'], format = "%Y-%m-%d")

    date_start = data_sales.date_of_day.max() - datetime.timedelta(21)
    df_train = data_sales[data_sales.date_of_day <= date_start]
    df_validation = data_sales[data_sales.date_of_day > date_start]
    print('\n')

    print("----------Training the model on all training data------------")
    model.fit(df_train, df_train.sold_qty_units, regressor__sample_weight=estimate_sample_weights(df_train, 5))
    y_train = np.clip(model.predict(df_train), 0, np.inf).astype(int)
    r2 = r2_score(df_train.sold_qty_units, y_train)
    print('\n')
    print('r2 score for the model fit', round(r2, 2))
    print('\n')

    print("------------Saving trained production model-------------")
    dump(model, args.output_model)
    print(f"Saved in {args.output_model}")
    print('\n')

    print("----------Generating predictions for the test data-------------")
    y_predicted = np.clip(model.predict(df_validation), 0, np.inf).astype(int)
    error_mae = round(mean_absolute_error(df_validation.sold_qty_units, y_predicted), 2)
    error_rmse = round(rmse(df_validation.sold_qty_units, y_predicted), 2)
    r2 = r2_score(df_validation.sold_qty_units, y_predicted)
    print('\n')
    
    print('MAE of the model trained = ', error_mae)
    print('RMSE of the model trained = ', error_rmse)
    print('r2 score for model prediction', round(r2, 2))
    
    df_test_results = pd.DataFrame({
        "art_no": df_validation.art_no,
        "date_of_day": df_validation.date_of_day,
        "sold_qty_units": y_predicted,
    })
    df_test_results.to_csv(args.output_test_pred, index=None)
    print(f"Saved predictions in {args.output_test_pred}")