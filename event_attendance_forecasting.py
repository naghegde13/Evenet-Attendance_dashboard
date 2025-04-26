# event_attendance_forecasting.py

"""
Event Attendance Forecasting and Geolocation Analytics
Full-Fledged Project Pipeline
"""

# --- Import Libraries ---
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

# --- 1. Simulate Event Data (Scraping Simulation) ---
def simulate_event_data(num_records=1000):
    event_types = ['In-person', 'Online', 'Hybrid']
    locations = ['Downtown', 'North York', 'Scarborough', 'Etobicoke', 'York']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    data = {
        'event_name': [f"Event_{i}" for i in range(num_records)],
        'event_type': np.random.choice(event_types, num_records),
        'event_start_hour': np.random.randint(6, 23, num_records),
        'event_day': np.random.choice(days_of_week, num_records),
        'event_location': np.random.choice(locations, num_records),
        'attendee_count': np.random.randint(10, 800, num_records),
        'latitude': np.random.uniform(43.6, 43.8, num_records),
        'longitude': np.random.uniform(-79.6, -79.2, num_records)
    }
    return pd.DataFrame(data)

# --- 2. Data Cleaning ---
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df

# --- 3. Exploratory Data Analysis (EDA) ---
def plot_eda(df):
    sns.histplot(df['attendee_count'], bins=30, kde=True)
    plt.title('Distribution of Attendee Counts')
    plt.xlabel('Number of Attendees')
    plt.ylabel('Frequency')
    plt.show()

    sns.countplot(data=df, x='event_day', order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    plt.title('Events by Day of the Week')
    plt.xticks(rotation=45)
    plt.show()

    sns.histplot(df['event_start_hour'], bins=17, kde=False)
    plt.title('Event Start Times')
    plt.xlabel('Hour of the Day')
    plt.show()

# --- 4. Heatmap of Event Locations ---
def generate_heatmap(df, output_html='event_heatmap.html'):
    m = folium.Map(location=[43.7, -79.4], zoom_start=11)
    for _, row in df.iterrows():
        folium.CircleMarker(location=[row['latitude'], row['longitude']],
                            radius=3, color='blue', fill=True).add_to(m)
    m.save(output_html)

# --- 5. Feature Engineering ---
def feature_engineering(df):
    bins = [0, 50, 200, 800]
    labels = ['Low', 'Medium', 'High']
    df['attendance_bucket'] = pd.cut(df['attendee_count'], bins=bins, labels=labels)
    df_encoded = pd.get_dummies(df[['event_type', 'event_start_hour', 'event_day']])
    return df, df_encoded

# --- 6. Machine Learning Model ---
def train_predict_attendance(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --- 7. Time Series Forecasting ---
def forecast_attendance():
    dates = pd.date_range(start='2024-01-01', periods=60)
    total_attendance = np.random.randint(3000, 6000, 60)
    ts_df = pd.DataFrame({'date': dates, 'total_attendance': total_attendance})

    X = np.arange(len(ts_df)).reshape(-1, 1)
    y = ts_df['total_attendance']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    future_days = np.arange(len(ts_df), len(ts_df) + 30).reshape(-1, 1)
    future_preds = model.predict(future_days)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(ts_df['date'], y, label='Historical')
    future_dates = pd.date_range(start=ts_df['date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
    plt.plot(future_dates, future_preds, label='Forecast', linestyle='--')
    plt.title('Attendance Forecast for Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Total Attendance')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print("\n🔵 Simulating Event Data...")
    df_events = simulate_event_data()

    print("\n🟢 Cleaning Data...")
    df_events = clean_data(df_events)

    print("\n🟡 Running Exploratory Data Analysis...")
    plot_eda(df_events)

    print("\n🔵 Generating Heatmap...")
    generate_heatmap(df_events)

    print("\n🟠 Feature Engineering...")
    df_events, df_encoded = feature_engineering(df_events)

    print("\n🟣 Training Attendance Prediction Model...")
    train_predict_attendance(df_encoded, df_events['attendance_bucket'])

    print("\n🟤 Forecasting Attendance (Time Series)...")
    forecast_attendance()

    print("\n✅ Full pipeline completed!")
