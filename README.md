# Stock Price Prediction using LSTM

This project predicts stock prices using an LSTM (Long Short-Term Memory) neural network. It fetches historical stock data using Yahoo Finance and predicts future stock prices based on trained models.

## Features
This project allows you to:
- Fetch stock data automatically using `yfinance`.
- Train an LSTM model for accurate stock price predictions.
- Visualize both historical and predicted stock prices with an intuitive graph.

## Project Structure
The project directory contains:
- `data_fetch.py`: The main script to fetch data, train the model, and make predictions.
- `requirements.txt`: A list of dependencies required to run the project.
- `README.md`: Documentation for understanding and using the project.

## Installation and Setup
To set up and run the project, follow these steps:

1. **Clone the repository**: First, download the project code from GitHub. You can do this by running:
   `git clone https://github.com/freddyziyuanwang/stock-price-prediction-lstm.git`  
   After cloning, navigate to the project directory by running:  
   `cd stock-price-prediction-lstm`

2. **Set up a virtual environment**: This helps to isolate the dependencies for the project. Create a virtual environment using:  
   `python3 -m venv env`  
   Then activate the virtual environment by running:  
   `source env/bin/activate`

3. **Install dependencies**: Install all required packages using the `requirements.txt` file. Run the following command:  
   `pip install -r requirements.txt`

## Usage
To use the project, run the `data_fetch.py` script by entering the command:  
`python3 data_fetch.py`  

The script will prompt you to input:
- **Stock ticker**: For example, type `AAPL` for Apple stock.
- **Start date**: Enter the historical data's start date in the format `YYYY-MM-DD` (e.g., `2020-01-01`).
- **End date**: Enter the end date for the historical data (e.g., `2023-01-01`).

The script will:
1. Fetch the historical stock data for the given ticker.
2. Train an LSTM model to predict stock prices.
3. Visualize the stock price prediction with a graph.

## Example Output
Below is an example of the prediction result generated by the project. The blue line represents historical prices, and the orange line shows the predicted prices:  

![Prediction Result](https://github.com/freddyziyuanwang/stock-price-prediction-lstm/blob/main/sample.png)

## Dependencies
The project requires the following Python packages:
- TensorFlow
- Pandas
- Matplotlib
- Scikit-learn
- Yfinance

These dependencies are listed in the `requirements.txt` file. If you have followed the installation instructions, they will be installed automatically.

## License
This project is licensed under the MIT License, allowing free use and modification. Refer to the `LICENSE` file for details.
