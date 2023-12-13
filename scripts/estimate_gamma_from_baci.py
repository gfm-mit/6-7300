import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_baci(fin, fout):
    df = pd.read_csv(fin)['i j v'.split()].groupby('i j'.split()).sum()
    df.to_csv(fout)


def ratio():
    y1 = pd.read_csv("baci2018.csv").set_index("i j".split())
    y2 = pd.read_csv("baci2019.csv").set_index("i j".split())
    yy = y1.join(y2, how="inner", lsuffix="_1", rsuffix="_2").reset_index()
    yy = yy.rename(columns=dict(i="x", j="m"))
    yy["log_x_ratio"] = np.log(yy.v_2 / yy.v_1)
    yy = yy[(yy.v_1 + yy.v_2) > 1e6]
    yy = yy.drop(columns="v_1 v_2".split())

    fx = pd.read_csv('forex.csv').query('TIME in [2018, 2019]')["LOCATION Value TIME".split()]
    fx = fx.set_index("LOCATION TIME".split()).unstack().dropna().loc[:, "Value"]
    fx["y_ratio"] = fx.loc[:, 2019] / fx.loc[:, 2018]
    fx = fx.drop(columns=[2018, 2019])

    cc = pd.read_csv("baci_cty.csv").set_index("iso_3digit_alpha").country_code.drop_duplicates().dropna()
    fx = fx.join(cc, how="inner")
    fx.country_code = fx.country_code.astype(int)
    fx = fx.set_index("country_code")

    yy = yy.join(fx, on="x", how="inner").rename(columns=dict(y_ratio="y_x_ratio"))
    yy = yy.join(fx, on="m", how="inner").rename(columns=dict(y_ratio="y_m_ratio"))
    yy = yy.set_index("x m".split())
    yy["log_y_ratio"] = np.log(yy.y_m_ratio / yy.y_x_ratio)
    yy = yy.drop(columns="y_x_ratio y_m_ratio".split())

    yy.log_x_ratio = np.clip(yy.log_x_ratio, -.2, .2)
    yy.log_y_ratio = np.clip(yy.log_y_ratio, -.1, .1)
    plt.scatter(yy.log_y_ratio, yy.log_x_ratio, alpha=0.1)
    plt.xlabel("delta log currency values")
    plt.ylabel("delta log exports")
    plt.title("does 2018->2019 change in exchange rate predict change in exports?")
    plt.show()