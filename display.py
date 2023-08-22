import pandas as pd
import matplotlib.pyplot as plt

pdir = "plots/"

df1 = pd.read_csv(pdir+"CNN1_history.csv")
df2 = pd.read_csv(pdir+"CNN2_history.csv")

plt.plot(df1["accuracy"], label="CNN1 train. acc.")
plt.plot(df1["val_accuracy"], label="CNN1 valid. acc.")
plt.plot(df2["accuracy"], label="CNN2 train. acc.")
plt.plot(df2["val_accuracy"], label="CNN2 valid. acc.")

plt.title("Training History")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()

plt.show()
