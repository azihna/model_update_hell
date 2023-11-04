# %%
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from mlxtend.evaluate import GroupTimeSeriesSplit
from scipy.stats import linregress
from sklearn import (
    base,
    compose,
    metrics,
    model_selection,
    pipeline,
    preprocessing,
)
from sklearn.experimental import enable_halving_search_cv  # noqa

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
        ("ohe", cat_enc),
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

search = model_selection.RandomizedSearchCV(
    full_pipeline,
    param_distributions,
    random_state=0,
    scoring="neg_mean_squared_error",
    cv=tscv.split(X, y, groups),
).fit(X, y)

initial_error = np.sqrt(-1 * search.best_score_)

# train the full model
model_full = base.clone(full_pipeline)
model_full.set_params(**search.best_params_)
model_full.fit(X, y)

#%%
st_range = pd.to_datetime(start_date.replace("_", "-")) + pd.offsets.DateOffset(1)
end_range = pd.to_datetime(end_date.replace("_", "-"))
date_range = pd.date_range(start=st_range, end=end_range)

# %%
update_errors = []
for dt in date_range:
    dt_str = str(dt).split()[0]
    update = pd.read_csv(
        f"data/updates/data_{dt_str}.csv",
        index_col=0,
        parse_dates=["order_date", "delivery_date"],
    )
    update.customer_id = update.customer_id.astype("object")
    update["target"] = (update.delivery_date - update.order_date).dt.days
    preds_update = model_full.predict(update)
    error = metrics.mean_squared_error(update["target"], preds_update, squared=False)
    update_errors.append(error)

# %%
idx = range(len(update_errors))
slope, intercept, _, _, _ = linregress(idx, update_errors)
trend_line = [slope * xi + intercept for xi in idx]

plt.plot(idx, update_errors, label="Data")
plt.plot(idx, trend_line, label="Trend Line", linestyle="--", color="green")
# Add the red line
plt.axhline(y=initial_error, color="red", linestyle="--", label="Estimated Error")

# Set labels and title
plt.xlabel("Date")
plt.ylabel("RMSE")
plt.title("Error over time")

# Add a legend
plt.legend()

# Show the plot
plt.show()

# %%
