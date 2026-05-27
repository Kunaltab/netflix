# Netflix Content Intelligence Dashboard

A Streamlit dashboard for exploring Netflix content data with filters, charts, country availability insights, and a TF-IDF recommendation engine.

## Project Files

- `app.py` - Streamlit dashboard application.
- `netflix_data.csv` - Dataset used by the dashboard and notebook.
- `Netflix.ipynb` - Exploratory analysis notebook.
- `requirements.txt` - Python packages needed to run the project.

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

The app expects `netflix_data.csv` to stay in the same folder as `app.py`.

## Features

- Filter by content type, release year, and genre.
- View key metrics for movies and series.
- Explore IMDb score distribution and release trends.
- See top countries by title availability.
- Generate recommendations based on genre and summary similarity.
