import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv("data/raw/spotify-tracks.csv")

# print(df["track_genre"].unique().tolist())

df = pd.read_csv("data/raw/df_all.csv")

print(df["artist_genres"].str.split(",").explode().str.strip().unique().tolist())
