import itertools
from datetime import datetime

import requests
import json
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import matplotlib.pyplot as plt

BEARER_TOKEN = "YOUR TOKEN fROM THINGSLOG PROFILE PAGE"

THINGSLOG_DEIVICE_ID ="01111111"
THINGSLOG_DEIVICE_ID_SENSOR_INDEX ="0"
FROM_DATE = "2022-01-12T00:00:00.000"
TO_DATE = "2022-02-12T00:00:00.000"
EVERY = 1
FREQ = "15min"

# Function to fetch data from the API
def fetch_data(api_url):
    try:
        headers = {
            'Authorization': f'Bearer {BEARER_TOKEN}',
            'Content-Type': 'application/json'
        }

        response = requests.get(api_url,headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None



def preprocess_data(data):
    # If 'data' is a JSON string, parse it into a Python object
    if isinstance(data, str):
        data = json.loads(data)

    time_series_data = []
    for item in data:
        value_array = item.get('value', [])
        for entry in value_array:
            date_str = entry.get('date')
            flow_value = entry.get('flow')

            if date_str and flow_value is not None:
                # Convert the date string to a datetime object
                try:
                    date_obj = datetime.fromisoformat(date_str)
                except ValueError:
                    # Handle invalid date strings, e.g., skip or log an error
                    continue

                flow_value = float(flow_value)
                time_series_data.append({
                    'date': date_obj,
                    'flow': flow_value
                })

    # Create a DataFrame from the processed data and set the index to 'date'
    df = pd.DataFrame(time_series_data).set_index('date')
    n = len(df)
    for i in range(n):
        current_flow = df.iloc[i]['flow']

        # Skip first and last entries to avoid index errors when accessing neighbors
        if i == 0 or i == n - 1:
            continue

        prev_flow = df.iloc[i - 1]['flow']
        next_flow = df.iloc[i + 1]['flow']
        avg_neighbors = (prev_flow + next_flow) / 2

        # Check for negative values and excessive high values
        if current_flow < 0 or current_flow > 10 * avg_neighbors:
            df.at[df.index[i], 'flow'] = avg_neighbors
    return df


# Function to perform SARIMA forecasting for one month ahead

def sarima_fit(series):
    # Define the parameter grid
    p = q = range(0, 3)
    d = range(0, 2)
    pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
    seasonal_pdq = [(x[0], x[1], x[2], 96) for x in list(itertools.product(p, d, q))]
    # Grid search
    min_aic = float('inf')
    best_params = None
    for param in pdq:
        print('Fitting param:', param)

        for seasonal_param in seasonal_pdq:
            print('Fitting param and seasonal_param:', param,seasonal_param)

            try:
                model = SARIMAX(series, order=param, seasonal_order=seasonal_param,freq='15min')
                results = model.fit()
                if results.aic < min_aic:
                    min_aic = results.aic
                    best_params = (param, seasonal_param)
                    print('Best SARIMA parameters:', best_params)
                    return best_params
            except:
                continue


def sarima_forecast(series):
    # Ensure the series has a datetime index and numeric values
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.set_index(pd.to_datetime(series.index))

    # Handle missing values by forward filling (you can choose another method as needed)
    series = series.fillna(method='ffill')

    try:
        # Check for non-numeric values
        # if not np.issubdtype(series.dtype, np.number):
        #     raise ValueError("Series contains non-numeric values")

        # Fit SARIMA model with appropriate order and seasonal_order
        # order = (1, 1, 1)
        # seasonal_order = (0, 0, 0, 12)  # Assuming monthly data
        order = (2, 1, 1)  # AR=2, Differencing=1, MA=1
        seasonal_order = (1, 0, 1, 96)  # Seasonal AR=1, Seasonal Differencing=0, Seasonal MA=1, m=96

 #       model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
 #       order = (2, 1, 1)  # AR=2, Differencing=1, MA=1
 #       model = ARIMA(series, order=order)

        # Check if the series has enough data points
        if len(series) < max(order[0], seasonal_order[3]):
            raise ValueError("Insufficient data for model parameters")
 #      best_order, best_seasonal_order = sarima_fit(series)

        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,freq={FREQ})

        results = model.fit(

        )
        # save model
        results.save('model.pkl')
        # load model
#        loaded = SARIMAXResults.load('model.pkl')
        # Forecast for the next month (adjust steps based on your data frequency)
        forecast_steps = 7*96  # For example, 30 months ahead
        forecast = results.get_forecast(steps=forecast_steps)
        return forecast

    except Exception as e:
        print(f"Error in SARIMA forecasting: {e}")
        return None



# Main script execution
if __name__ == "__main__":
    # Replace with your API endpoint
    api_url = f"https://iot.thingslog.com:4443/api/devices/{THINGSLOG_DEIVCE_ID}/${THINGSLOG_SENSOR_INDEX}/flows?every={EVERY}&fromDate={FROM_DATE}&toDate={TO_DATE}"


    # Fetch data from the API
    json_data = fetch_data(api_url)

    if json_data:
        # Preprocess the data
        series = preprocess_data(json_data)

        # Perform SARIMA forecasting
        forecast = sarima_forecast(series)

        if forecast is not None:
            # Get confidence intervals
            pred_ci = forecast.conf_int()

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(series, label='Observed')
            plt.plot(forecast.predicted_mean, label='Forecast', color='red')
            plt.fill_between(pred_ci.index,
                             pred_ci.iloc[:, 0],
                             pred_ci.iloc[:, 1], color='pink', alpha=0.1)
            plt.title('SARIMA Forecast of Flow Data for Next Week')
            plt.xlabel('Timestamp')
            plt.ylabel('Flow Value')
            plt.legend()
            plt.show()
