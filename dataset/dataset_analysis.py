# %%

import pandas as pd
from sklearn.datasets import fetch_california_housing

# Wczytanie danych
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['MedHouseVal'] = california_housing.target

# 1. Mediana wartości domów w obszarach o wysokim medianowym dochodzie
high_income = df[df['MedInc'] > df['MedInc'].quantile(0.75)]
print(high_income['MedHouseVal'].mean())

# 2. Zależność między wiekiem domu a medianą wartości domów
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x='HouseAge', y='MedHouseVal', data=df)
plt.show()

# 3. Zależność między liczbą pokoi a medianą wartości domów
sns.scatterplot(x='AveRooms', y='MedHouseVal', data=df)
plt.show()

# 4. Zależność między populacją a medianą wartości domów
sns.scatterplot(x='Population', y='MedHouseVal', data=df)
plt.show()

# 5. Zależność między lokalizacją geograficzną a medianą wartości domów
sns.scatterplot(x='Longitude', y='Latitude', hue='MedHouseVal', data=df)
plt.show()