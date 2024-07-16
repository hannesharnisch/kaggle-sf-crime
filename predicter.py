from os import path
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import re
import time
import argparse
from astral import LocationInfo
import joblib
import numpy as np
from datetime import datetime
from astral.sun import sun
import pandas as pd
import pytz
from sklearn.discriminant_analysis import StandardScaler

# Funktion zur Validierung des Datums


def validate_date(input: str) -> bool:
    try:
        if input:
            pd.to_datetime(input)
        return True
    except ValueError:
        return False

# Funktion zur Validierung von x und y (Beispiel)


def validate_number(input: str, min: int = None, max: int = None):
    try:
        zahl = float(input)
        if (min is not None and zahl < min) or (max is not None and zahl > max):
            return False
        return True
    except ValueError:
        return False


def validate_model(input: str):
    if not input:
        return False
    return input in ['1', '2', '3']

    ############################################################################################################


def encode(encoded_train):
    # Transforming Categorical attributes -> Nummerical Attributes according to Slide 25 Data Perparation
    def create_columns_for_unique_values(train_df, column, unique_values):
        for value in unique_values:
            train_df[column + "-" +
                     value] = (train_df[column] == value).astype(int)

    columns = {
        'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'PdDistrict': ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN'],
        'StreetType': ['AV', 'ST', 'CT', 'PZ', 'LN', 'DR', 'PL', 'HY', 'WY', 'TR', 'RD', 'BL', 'WAY', 'CR', 'AL', 'I-80', 'RW', 'WK', 'INT', 'OTHER'],
        'Season': ['Winter', 'Spring', 'Summer', 'Fall']
    }

    for column, unique_values in columns.items():
        create_columns_for_unique_values(encoded_train, column, unique_values)

    encoded_train.drop(columns=columns, inplace=True)

    scaler = StandardScaler()
    encoded_train[['lat', 'long']] = scaler.fit_transform(
        encoded_train[['lat', 'long']])

    encoded_train.head()

    return encoded_train


def get_dataframe(x: float, y: float, date: str, district: str, address: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "Dates": [pd.to_datetime(date)],
        "lat": [y],
        "long": [x],
        "Address": [address],
        "PdDistrict": [district]
    })

    add_datetime_parts(df)
    add_holiday_feature(df)
    add_night_feature(df)
    add_season_feature(df)
    add_address_feature(df)

    df['Season'] = df['Season'].astype('category')
    df['StreetType'] = df['StreetType'].astype('category')

    sorted_column_sequence = ['DayOfWeek', 'Day', 'Month', 'Hour', 'Minute', 'Season', 'Night',
                              'Holiday', 'Block', 'StreetType', 'PdDistrict', 'lat', 'long']
    df = df[sorted_column_sequence]
    df = encode(df)

    return df


def add_holiday_feature(df: pd.DataFrame):
    cal = calendar()
    holidays = cal.holidays(start=df['Dates'].min(), end=df['Dates'].max())
    df['Holiday'] = (df['Dates'].dt.date.astype(
        'datetime64[ns]').isin(holidays)).astype(int)


def add_datetime_parts(df: pd.DataFrame):
    df['DayOfWeek'] = df['Dates'].dt.dayofweek.map(
        {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
    for part in ["year", "month", "day", "hour", "minute"]:
        create_column(df, "Dates", part)


def add_night_feature(df: pd.DataFrame):
    unique_days = df[['Day', 'Month', 'Year']].drop_duplicates()
    sun_info = get_all_sunset_sunrise_sf(unique_days)
    df['Night'] = df['Dates'].map(lambda x: int(
        is_at_night(x, sun_info[f"{x.day}-{x.month}-{x.year}"])))


def add_season_feature(df: pd.DataFrame):
    seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df['Season'] = df['Month'].map(lambda x: seasons[(x % 12 + 3)//3])


def add_address_feature(df: pd.DataFrame):
    df['Block'] = df['Address'].map(get_block)
    df['StreetType'] = df['Address'].map(get_street_type)


def get_block(address):
    match = re.search(r'(\d+)\s+block of', address, re.IGNORECASE)
    if match:
        # The block number is divided by 100 because they always increase by 100 and then increased by 1 to leave 0 for no block
        return int(match.group(1)) // 100 + 1
    return 0


def get_street_type(address):
    # See also data-understanding.ipynb
    street_types = ['AV', 'ST', 'CT', 'PZ', 'LN', 'DR', 'PL', 'HY', 'FY',
                    'WY', 'TR', 'RD', 'BL', 'WAY', 'CR', 'AL', 'I-80', 'RW', 'WK']
    match = re.findall(r'\b(?:' + '|'.join(street_types) +
                       r')\b', address, re.IGNORECASE)
    if len(match) > 1 and '/' in address:
        return "INT"
    if len(match) == 1:
        return match[0]
    return "OTHER"


def create_column(df, datetime_column, part_name):
    df[part_name.capitalize()] = df[datetime_column].map(
        lambda x: getattr(x, part_name, None))


def get_all_sunset_sunrise_sf(x):
    city = LocationInfo("San Francisco", "USA",
                        "America/Los_Angeles", 37.7749, -122.4194)
    timezone = pytz.timezone(city.timezone)
    return {
        f"{day['Day']}-{day['Month']}-{day['Year']}": sun(
            city.observer,
            date=pd.Timestamp(
                year=day['Year'], month=day['Month'], day=day['Day'], tz=timezone).date(),
            tzinfo=city.timezone
        )
        for _, day in x.iterrows()
    }


def is_at_night(date, sun_info):
    dusk = sun_info['dusk'].replace(tzinfo=None)
    dawn = sun_info['dawn'].replace(tzinfo=None)
    # Keine Änderung hier, da die Logik korrekt ist, aber stellen Sie sicher, dass 'date' auch ohne Zeitzone ist
    if dawn < dusk:  # Für Fälle, in denen der Sonnenaufgang als am nächsten Tag betrachtet wird
        return date > dusk or date < dawn
    else:
        return dusk < date < dawn


def get_district_from_coordinates(lat, long):
    import geopandas as gpd
    from shapely.geometry import Point
    # Load the GeoJSON file
    gdf = gpd.read_file('./data/sf-districts.json')
    point_of_interest = Point(long, lat)

    # Find the district containing the point
    district = gdf[gdf.geometry.contains(point_of_interest)]['DISTRICT']

    if not district.empty:
        return district.iloc[0]
    else:
        return None


MODELS = {
    1: ("Decision Tree", './models/decision_tree/decision_tree.pkl'),
    2: ("Random Forest", './models/random_forest/random_forest_model.pkl'),
    3: ("XGBoost", './models/xgboost/xgboost.pkl', './models/xgboost/label_encoder.pkl')
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='SF-Crime Predicter',
        description='This program uses different models to predict the category of a crime in San Francisco.')

    parser.add_argument('-m', '--model', type=int, choices=[1, 2, 3], default=1,
                        help='Model to use: 1 (Decision Tree), 2 (Random Forest), 3 (XGBoost)')
    parser.add_argument('-t', '--time', type=str, default=datetime.today(),
                        help='Date and time of the crime')
    parser.add_argument('-a', '--address', nargs="*", type=str,
                        help='Address of the location')
    parser.add_argument('--long', type=float,
                        help='Longitude of the location (SF: -123 - -122.3)')
    parser.add_argument('--lat', type=float,
                        help='Latitude of the location (SF: 37.7 - 37.8)')
    parser.add_argument('-d', '--district', nargs='?', type=str,
                        choices=['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN',
                                 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN'],
                        help='District of the Crime')
    parser.add_argument('-lm', '--list-models',
                        action='store_true', help='List all available models')

    args = parser.parse_args()

    if args.list_models:
        print(f"{'Id':<4}{'Name':<15}{'Available':<10}")
        print("-" * 40)
        for id, model in MODELS.items():
            print(f"{id:<4}{model[0]:<15}{
                  ('YES' if path.exists(model[1]) else 'NO'):<10}")
        exit()

    # Benutzereingaben erfassen mit Validierung
    while args.long is None:
        args.long = input("Please enter the longitude: ")
        if validate_number(args.long, min=-123, max=-122.3):
            args.long = float(args.long)
            break
        print("Invalid Entry. Please provide a number (SF: -123 - -122.3)")

    while args.lat is None:
        args.lat = input("Please enter the latitude: ")
        if validate_number(args.lat, min=37.7, max=37.8):
            args.lat = float(args.lat)
            break
        print("Invalid Entry. Please provide a number (SF: 37.7 - 37.8)")

    # Beispiel für eine einfache Texteingabe ohne Validierung
    if not args.district:
        print("Trying to look up district from coordinates...")
        args.district = get_district_from_coordinates(args.lat, args.long)
        if not args.district:
            print(
                "District could not be determined. Please try another location or enter the district manually.")
            exit(0)
        print(f"Location is in District: {args.district}")

    address = " ".join(args.address)
    if not address:
        address = input("Bitte geben Sie die Addresse ein: ")

    model = joblib.load(MODELS[args.model][1])

    print("\nPreparing Data for Prediction...")
    df = get_dataframe(args.long, args.lat, args.time,
                       args.district, address)
    df = df[model.feature_names_in_]
    print("Starting Prediction...\n")
    start_time = time.time()
    # Vorhersage durchführen mit DataFrame
    prediction = model.predict_proba(df)

    end_time = time.time()
    prediction_time = end_time - start_time

    target_categories = model.classes_
    mapped_predictions = dict(zip(target_categories, prediction[0]))
    sorted_predictions = sorted(
        mapped_predictions.items(), key=lambda x: x[1], reverse=True)
    top_3_predictions = sorted_predictions[:3]
    print("\nTop 3 Predictions:\n")
    print(f"{'Category':<20}{'Confidence':<10}")
    print("-" * 30)
    for category, probability in top_3_predictions:
        # Use label encoder for XGBoost
        if args.model == 3:
            label_encoder = joblib.load(MODELS[args.model][2])
            decoded_cat = label_encoder.inverse_transform([category])[0]
            print(f"{decoded_cat:<20}{probability:.2f}")
        else:
            print(f"{category:<20}{probability:.2f}")

    print(f"\nPrediction time: {prediction_time:.4f} seconds")
