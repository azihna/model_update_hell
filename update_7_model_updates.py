# %%
import copy

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from mlflow.models import infer_signature
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

client = mlflow.MlflowClient()

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

# %%
mlflow.set_tracking_uri("mlruns")

model_name = f"lgbm_{start_date}"

with mlflow.start_run(run_name=f"model_{start_date}") as run:
    model_full = base.clone(full_pipeline)
    model_full.set_params(**search.best_params_)
    model_full.fit(X, y)
    preds = model_full.predict(X)

    signature = infer_signature(X, preds)
    mlflow.sklearn.log_model(model_full, model_name, signature=signature)

    param_dict = {
        key.replace("regressor__", ""): value
        for (key, value) in search.best_params_.items()
    }
    mlflow.log_params(param_dict)
    mlflow.log_metric("rmse", initial_error)

    run_id = run.info.run_id

# %%

reg_name = "delivery-estimator"

result = mlflow.register_model(f"runs:/{run_id}/{model_name}", "delivery-estimator")

client.transition_model_version_stage(
    name=reg_name,
    version=1,
    stage="Production",
)

#%%
st_range = pd.to_datetime(start_date.replace("_", "-")) + pd.offsets.DateOffset(1)
end_range = pd.to_datetime(end_date.replace("_", "-"))
date_range = pd.date_range(start=st_range, end=end_range)

stage = "Production"

# %%
update_errors = []
df_with_updates = df.copy(deep=True)
for dt in date_range:

    model_prod = mlflow.pyfunc.load_model(model_uri=f"models:/{reg_name}/{stage}")

    dt_str = str(dt).split()[0]
    update = pd.read_csv(
        f"data/updates/data_{dt_str}.csv",
        index_col=0,
        parse_dates=["order_date", "delivery_date"],
    )
    update.customer_id = update.customer_id.astype("object")
    update["target"] = (update.delivery_date - update.order_date).dt.days
    preds_update = model_prod.predict(update)
    error = metrics.mean_squared_error(update["target"], preds_update, squared=False)
    update_errors.append(error)
    upt_rolling = pd.Series(update_errors).rolling(window=7, min_periods=1).mean()

    trigger_rolling = upt_rolling.iloc[-1] > 2.8

    preds_update = pd.Series(preds_update, index=update.index, name="preds")
    df_pred = pd.concat((update, preds_update), axis=1)

    customer_error = (
        df_pred[df_pred.customer_status == "platinum"]
        .groupby("customer_status")
        .apply(
            lambda x: metrics.mean_squared_error(x["preds"], x["target"], squared=False)
        )
    )
    customer_trigger = (
        customer_error.iloc[-1] > 4 if customer_error.shape[0] > 0 else False
    )

    if trigger_rolling or customer_trigger:
        print(f"Updating the model on {dt}")
        # update with the latest data, check the error again
        # if lower now go ahead, else throw an error

        # normally you'd go over the normal train test set again and then
        # fit and compare the day again but for simplicity we'll
        # just compare the update's error
        with mlflow.start_run(run_name=f"model_{dt}") as run:
            # first try updating the model with new data
            model_updated = copy.deepcopy(model_prod._model_impl.sklearn_model)
            X_upt = df_with_updates.drop(
                ["target", "order_date", "delivery_date"], axis=1
            )
            y_upt = df_with_updates["target"]
            group_upt = df_with_updates["order_date"]
            model_updated.fit(X_upt, y_upt)
            preds_with_new_model = model_updated.predict(update)
            error_new = metrics.mean_squared_error(
                update["target"], preds_with_new_model, squared=False
            )
            model_upt_name = f"lgbm_{dt}"
            mlflow.sklearn.log_model(model_updated, model_upt_name, signature=signature)
            param_dict = {
                key.replace("regressor__", ""): value
                for (key, value) in search.best_params_.items()
            }
            mlflow.log_params(param_dict)
            mlflow.log_metric("rmse", error_new)
            result = mlflow.register_model(
                f"runs:/{run.info.run_id}/{model_upt_name}", "delivery-estimator"
            )
            version = client.get_latest_versions("delivery-estimator", stages=["None"])[
                0
            ].version
            client.transition_model_version_stage(
                name=reg_name,
                version=version,
                stage="Staging",
            )
            if error_new > error:
                version_prod = client.get_latest_versions(
                    "delivery-estimator", stages=["Production"]
                )[0].version
                client.transition_model_version_stage(
                    name=reg_name,
                    version=version_prod,
                    stage="Archived",
                )
                client.transition_model_version_stage(
                    name=reg_name,
                    version=version,
                    stage="Production",
                )
            else:
                # if not try doing another hyperparameter search
                print("Raise error and alert the users.")
                client.transition_model_version_stage(
                    name=reg_name,
                    version=version,
                    stage="Archived",
                )
    else:
        print(f"No update needed on date {dt}")

    df_with_updates = pd.concat((df_with_updates, update))


# %%
idx = range(len(update_errors))
slope, intercept, _, _, _ = linregress(idx, update_errors)
trend_line = [slope * xi + intercept for xi in idx]

plt.plot(idx, update_errors, label="Data")
plt.plot(idx, trend_line, label="Trend Line", linestyle="--", color="green")
plt.plot(idx, upt_rolling, label="Rolling Error", linestyle="dotted", color="orange")
# Add the red line
plt.axhline(y=initial_error, color="red", linestyle="dashdot", label="Estimated Error")

# Set labels and title
plt.xlabel("Date")
plt.ylabel("RMSE")
plt.title("Error over time")

# Add a legend
plt.legend()

# Show the plot
plt.show()

# %%
