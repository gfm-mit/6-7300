import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#g = pd.read_csv("wb_gdp.csv")["c g".split()].set_index("c").g.drop_duplicates()
#tot = g.loc["WLD"]
#cc = pd.read_csv("baci_country_codes.csv").set_index("iso_3digit_alpha").country_code.drop_duplicates()
#
#gg = g.to_frame().join(cc.to_frame()).dropna()
#gg.country_code = gg.country_code.astype(int)
#gg = gg.set_index("country_code")

#x = pd.read_csv("baci2019.csv")
#x.i = x.i.astype(int)
#x.j = x.j.astype(int)
#x["i_gdp"] = x.i.map(gg.g)
#x["j_gdp"] = x.j.map(gg.g)
#x["scaled"] = x.i_gdp * x.j_gdp / tot / (1e3 * x.v)
#x["log_scaled"] = np.log(x.scaled)
##plt.hist(np.log(g), bins=30)
##plt.title("{} : {}".format(np.std(np.log(g)), np.nanmedian(np.log(g))))
#plt.hist(x.log_scaled, bins=30)
#plt.title("{} : {}".format(np.std(x.log_scaled), np.nanmedian(x.log_scaled)))
##plt.title(np.nanmedian(np.log10(x.log_scaled)))
#plt.show()

fx = pd.read_csv('forex.csv').query('TIME in [2018, 2019]')["LOCATION Value TIME".split()]
fx = fx.set_index("TIME LOCATION".split()).sort_index()
##"LOCATION","INDICATOR","SUBJECT","MEASURE","FREQUENCY","TIME","Value","Flag Codes"
print(fx)

def get_baci(year):
    x = pd.read_csv(f"baci{year}.csv")
    f = fx.loc[year]