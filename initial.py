# %%
import pandas as pd

# %%
df = pd.read_csv(
    "data/raw/orders.csv",
    parse_dates=["order_date", "delivery_date"],
    date_format="%d-%b-%y"
)

df = df[df.order_date != df.delivery_date]
df.customer_status = df.customer_status.str.lower()

df_supp = pd.read_csv("data/raw/product-supplier.csv")

df_merged = pd.merge(df, df_supp, on="product_id")
df_merged = df_merged.drop(
    ["supplier_id", "order_id", "product_id"],
    axis=1
)

# %%
# how many orders per month in general?
df_merged = df_merged[
    df_merged.order_date.between(pd.to_datetime("2019-10-01"),
                                 pd.to_datetime("2021-10-01")
                                 )
    ]
df_merged.to_csv("data/full_data/joined_data.csv")

# %%
# initial available data
df_init = df_merged[
    df_merged.order_date.between(
        pd.to_datetime("2019-10-01"),
        pd.to_datetime("2021-04-01")
        )
]

df_init.to_csv("data/initial/data_2021_04_01.csv")

# %%
df_daily = df_merged[
    df_merged.order_date > pd.to_datetime("2021-04-01")
]
date_list = df_daily.order_date.unique().tolist()

for dt in date_list:
    df_filt = df_daily[
        df_daily.order_date == dt
    ]
    df_filt.to_csv(
        "data/updates/data" + str(dt).split()[0] + ".csv"
    )


# %%
