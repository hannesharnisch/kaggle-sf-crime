from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import re
from astral import LocationInfo
import joblib
import numpy as np
from datetime import datetime
from astral.sun import sun

import pandas as pd
import pytz

# Funktion zur Validierung des Datums


def validate_date(input: str) -> bool:
    try:
        if input:
            datetime.strptime(input, '%Y-%m-%d')
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
    return input in ['1', '2', '3']

############################################################################################################


def get_dataframe(x: float, y: float, date: str, district: str) -> pd.DataFrame:
    data = {
        'long': [x],
        'lat': [y],
        'Block': None,
        'Day': None,
        'DayOfWeek-Friday': None,
        'DayOfWeek-Monday': None,
        'DayOfWeek-Saturday': None,
        'DayOfWeek-Sunday': None,
        'DayOfWeek-Thursday': None,
        'DayOfWeek-Tuesday': None,
        'DayOfWeek-Wednesday': None,
        'Holiday': None,
        'Hour': None,
        'Minute': None,
        'Month': None,
        'Night': None,
        'PdDistrict-BAYVIEW': None,
        'PdDistrict-CENTRAL': None,
        'PdDistrict-INGLESIDE': None,
        'PdDistrict-MISSION': None,
        'PdDistrict-NORTHERN': None,
        'PdDistrict-PARK': None,
        'PdDistrict-RICHMOND': None,
        'PdDistrict-SOUTHERN': None,
        'PdDistrict-TARAVAL': None,
        'PdDistrict-TENDERLOIN': None,
        'Season-Fall': None,
        'Season-Spring': None,
        'Season-Summer': None,
        'Season-Winter': None,
        'StreetType-AL': None,
        'StreetType-AV': None,
        'StreetType-BL': None,
        'StreetType-CR': None,
        'StreetType-CT': None,
        'StreetType-DR': None,
        'StreetType-HY': None,
        'StreetType-I-80': None,
        'StreetType-INT': None,
        'StreetType-LN': None,
        'StreetType-OTHER': None,
        'StreetType-PL': None,
        'StreetType-PZ': None,
        'StreetType-RD': None,
        'StreetType-RW': None,
        'StreetType-ST': None,
        'StreetType-TR': None,
        'StreetType-WAY': None,
        'StreetType-WK': None,
        'StreetType-WY': None,
    }

    df = pd.DataFrame(data)

    return df
    df["Dates"] = pd.to_datetime(df["Dates"], format="%Y-%m-%d")

    add_datetime_parts(df)
    add_holiday_feature(df)
    add_night_feature(df)
    add_season_feature(df)
    add_address_feature(df)

    df['Season'] = df['Season'].astype('category')
    df['StreetType'] = df['StreetType'].astype('category')
    return df


def add_holiday_feature(df: pd.DataFrame):
    cal = calendar()
    holidays = cal.holidays(start=df['Dates'].min(), end=df['Dates'].max())
    df['Holiday'] = (df['Dates'].dt.date.astype(
        'datetime64[ns]').isin(holidays)).astype(int)


def add_datetime_parts(df: pd.DataFrame):
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


if __name__ == "__main__":

    # Benutzereingaben erfassen mit Validierung
    while True:
        x = input("Bitte geben Sie x (long) ein: ")
        if validate_number(x):
            x = float(x)
            break
        print("Ungültige Eingabe. Bitte geben Sie eine Zahl ein.")

    while True:
        y = input("Bitte geben Sie y (lat) ein: ")
        if validate_number(y, min=0):
            y = float(y)
            break
        print("Ungültige Eingabe. Bitte geben Sie eine gültige Zahl größer als 0 ein.")

    while True:
        date = input(
            "Bitte geben Sie das Datum ein (z.B. 2023-04-01) oder drücken Sie Enter für das heutige Datum: ")
        if not date or validate_date(date):
            date = date or datetime.today().strftime('%Y-%m-%d')
            break
        print("Ungültige Eingabe. Bitte geben Sie das Datum im Format YYYY-MM-DD ein.")

    # Beispiel für eine einfache Texteingabe ohne Validierung
    district = input("Bitte geben Sie den Bezirk ein: ")

    while True:
        model = input(
            "Bitte geben Sie ein Modell ein: 1 (Decision Tree), 2 (Random Forest), 3 (Gradient Boosting): ")
        if not model or validate_model(model):
            print(model)
            if model == '1':
                model = joblib.load('./models/decision_tree/decision_tree.pkl')
            break
        print("Ungültige Eingabe. Bitte geben Sie ein gültiges Modell ein")

    df = get_dataframe(x, y, date, district)
    print(df.head())
    # Vorhersage durchführen mit DataFrame
    prediction = model.predict(df)
