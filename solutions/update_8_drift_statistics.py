# %%
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift, LabelDrift
from deepchecks.tabular.suites import train_test_validation
from mlxtend.evaluate import GroupTimeSeriesSplit
from sklearn import compose, pipeline, preprocessing

# %%

start_date = "2021_04_01"
end_date = "2021_10_01"

df = pd.read_csv(
    f"data/initial/data_{start_date}.csv",
    index_col=0,
    parse_dates=["order_date", "delivery_date"],
)
df.customer_id = df.customer_id.astype("object")
df["target"] = (df.delivery_date - df.order_date).dt.days

df = df.sort_values("order_date")
groups = df["order_date"]

X = df.drop(["target", "order_date", "delivery_date"], axis=1)
y = df["target"]

numerical_columns = X.select_dtypes(exclude="object").columns
categorical_columns = X.select_dtypes(include="object").columns

# add category encoder
cat_enc = CatBoostEncoder()
scaler = preprocessing.StandardScaler()

# numeric variables preprocessing pipelines
numeric_pipeline = pipeline.Pipeline([("scaler", scaler)])

# categorical variables  preprocessing pipelines
categorical_pipeline = pipeline.Pipeline(
    [
        ("catboost", cat_enc),
    ]
)

# combine pipelines
composed = compose.ColumnTransformer(
    [
        ("num", numeric_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ]
)

model = lgb.LGBMRegressor(verbosity=-1)

# final pipeline
full_pipeline = pipeline.Pipeline([("col_trans", composed), ("regressor", model)])


# %%
tscv_args = {"n_splits": 3, "test_size": 30}
tscv = GroupTimeSeriesSplit(**tscv_args)

param_distributions = {
    "regressor__max_depth": [2, 5, 7],
    "regressor__num_iterations": [100, 250, 500],
    "regressor__lambda_l1": [0, 0.25, 0.5, 0.75],
    "regressor__num_leaves": [64, 128, 256],
    "regressor__learning_rate": [0.01, 0.1, 0.2, 0.3],
}


tr_idx, te_idx = next(tscv.split(X, y, groups))
df_train = df.drop(["order_date", "delivery_date"], axis=1).iloc[tr_idx, :]
df_test = df.drop(["order_date", "delivery_date"], axis=1).iloc[te_idx, :]


train_dataset = Dataset(df_train, label="target", cat_features=categorical_columns)
full_pipeline.fit(
    train_dataset.data[train_dataset.features],
    train_dataset.data[train_dataset.label_name],
)

test_dataset = Dataset(df_test, label="target", cat_features=categorical_columns)

# %%
check = FeatureDrift()
result = check.run(
    train_dataset=train_dataset, test_dataset=test_dataset, model=full_pipeline
)
result.show()
# %%
check = LabelDrift()
result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
result
# %%
validation_suite = train_test_validation()
suite_result = validation_suite.run(train_dataset, test_dataset)
suite_result

# %%
(suite_result.get_not_passed_checks()[0].conditions_results[0].category.value)
# %%
