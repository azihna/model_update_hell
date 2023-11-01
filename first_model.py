# %%
import pandas as pd
from sklearn import linear_model, metrics
from category_encoders import CatBoostEncoder


# %%

df = pd.read_csv(
    "data/initial/data_2021_04_01.csv",
    index_col=0,
    parse_dates=["order_date", "delivery_date"],
)

# %%

df["target"] = (df.delivery_date - df.order_date).dt.days

# %%
X = df.drop(
    ["target", "order_date", "delivery_date"],
    axis=1
)
y = df["target"]

# add category encoder
cat_enc = CatBoostEncoder()
X = cat_enc.fit_transform(X, y)


# %% 
# fit the model
model = linear_model.LinearRegression()
model.fit(X, y)


# %%
# add a metric
rmse = metrics.mean_squared_error(y, model.predict(X), squared=False)


# %%
# simple cross validation
# add MLFlow
