import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import random
from datetime import datetime, timedelta

# Generate synthetic data
np.random.seed(42)
n_samples = 5000

# Generate timestamps within a day
base_date = datetime(2024, 1, 1)
timestamps = [base_date + timedelta(minutes=random.randint(0, 24*60-1)) for _ in range(n_samples)]

# Generate coordinates (assuming city area)
coordinates_x = np.random.uniform(0, 10, n_samples)  # 10km x 10km area
coordinates_y = np.random.uniform(0, 10, n_samples)

# Generate vehicle types
vehicle_types = np.random.choice(['motorcycle', 'car', 'bus'], n_samples, p=[0.6, 0.3, 0.1])

# Generate average speeds (km/h)
speeds = np.random.normal(30, 10, n_samples)
speeds = np.clip(speeds, 5, 60)  # Limit speeds between 5 and 60 km/h

# Generate traffic density
traffic_density = np.random.choice(['low', 'medium', 'high'], n_samples)

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'coord_x': coordinates_x,
    'coord_y': coordinates_y,
    'vehicle_type': vehicle_types,
    'speed': speeds,
    'traffic_density': traffic_density
})

# Extract hour from timestamp
df['hour'] = df['timestamp'].dt.hour

# Define peak hours (e.g., 7-9 AM and 4-6 PM)
df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 18))
df['is_peak_hour'] = df['is_peak_hour'].astype(int)

# Calculate large vehicle ratio
df['is_large_vehicle'] = (df['vehicle_type'].isin(['car', 'bus'])).astype(int)
large_vehicle_ratio = df.groupby(['coord_x', 'coord_y'])['is_large_vehicle'].mean().reset_index()
df = df.merge(large_vehicle_ratio, on=['coord_x', 'coord_y'], suffixes=('', '_ratio'))

# Identify congestion points using K-means
X_cluster = df[['coord_x', 'coord_y', 'hour']]
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Calculate congestion severity index
df['congestion_severity'] = ((df['traffic_density'] == 'high').astype(int) * 3 + 
                           (df['traffic_density'] == 'medium').astype(int) * 2 + 
                           (df['traffic_density'] == 'low').astype(int)) * (60 - df['speed']) / 60

# Prepare data for traffic density prediction
X = df[['coord_x', 'coord_y', 'hour', 'is_peak_hour', 'is_large_vehicle_ratio', 'congestion_severity']]
y = df['traffic_density']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Print model accuracy
print(f"Model accuracy: {gb_model.score(X_test, y_test):.2f}")

# Save the data to CSV
df.to_csv(r'E:\HOC\Lập trình Python\baitapvenha\bai_thuc_hanh_5_5\bai3\Data_Number_6.csv', index=False)
