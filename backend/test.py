import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}"
)

print(engine.url)
df = pd.read_sql("""
SELECT *
FROM recent_tracks_audio_features
""", engine)
print(df)

df.isna().mean().sort_values(ascending=False)
