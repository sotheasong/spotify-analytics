# analysis.py
import pandas as pd
import plotly.express as px
from collections import Counter

def genre_chart(top_artists_json):
    # Flatten genres
    genres = []
    for artist in top_artists_json:
        genres.extend(artist.get("genres", []))

    # Count genres
    genre_counts = Counter(genres)
    df = pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"])
    df = df.sort_values(by="Count", ascending=False).head(10)

    # Create interactive Plotly chart
    fig = px.bar(
        df,
        x="Count",
        y="Genre",
        orientation="h",
        title="🎵 Your Top 10 Genres",
        color="Count",
        color_continuous_scale="Viridis"
    )

    # Convert figure to HTML
    return fig.to_html(full_html=False)
