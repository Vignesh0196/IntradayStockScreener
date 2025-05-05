import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class StockAnalyzer:
    def __init__(self, csv_file_path=None):
        """
        Initialize the Stock Analyzer with optional CSV file path
        
        Parameters:
        - csv_file_path (str, optional): Path to the CSV file containing historical data
        """
        self.data = None
        self.original_data = None  # Initialize original_data
        self.current_interval = '5min'  # Default interval
        self.filtered_by_date = False
        
        if csv_file_path:
            self.load_data(csv_file_path)
    
    def load_data(self, csv_file_path):
        """
        Load data from a CSV file
        
        Parameters:
        - csv_file_path (str): Path to the CSV file
        
        Returns:
        - bool: True if successful, False otherwise
        """
        try:
            # Load the CSV file
            data = pd.read_csv(csv_file_path)
            
            # Check if required columns exist
            required_columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    print(f"Error: Required column '{col}' not found in CSV.")
                    return False
            
            # Convert date and time columns to datetime and set as index
            data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
            data.set_index('datetime', inplace=True)
            
            # Ensure numeric columns are of the right type
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Drop rows with NaN values in critical columns
            data.dropna(subset=numeric_cols, inplace=True)
            
            self.data = data
            self.original_data = data.copy()  # Keep a copy of the original data
            
            print(f"Successfully loaded data with {len(data)} rows")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def filter_by_date(self, date_str):
        """
        Filter the data to show only records for a specific date
        
        Parameters:
        - date_str (str): Date string in 'YYYY-MM-DD' format
        
        Returns:
        - bool: True if successful, False otherwise
        """
        if self.original_data is None:
            print("No data loaded. Please load data first.")
            return False
            
        try:
            # Parse the date string
            target_date = pd.to_datetime(date_str).date()
            
            # Filter the data to include only the specified date
            filtered_data = self.original_data[self.original_data.index.date == target_date]
            
            if filtered_data.empty:
                print(f"No data found for date: {date_str}")
                return False
                
            # Update the current data with the filtered data
            self.data = filtered_data
            self.filtered_by_date = True
            
            print(f"Data filtered to show only {date_str} with {len(filtered_data)} records")
            return True
            
        except Exception as e:
            print(f"Error filtering data by date: {e}")
            return False
    
    def reset_filter(self):
        """
        Reset any filters and return to the original data
        
        Returns:
        - bool: True if successful, False otherwise
        """
        if self.original_data is None:
            print("No data loaded. Please load data first.")
            return False
            
        self.data = self.original_data.copy()
        self.filtered_by_date = False
        print(f"Filters reset. Using all {len(self.data)} records.")
        return True
    
    def resample_data(self, interval):
        """
        Resample the data to a different time interval
        
        Parameters:
        - interval (str): Target interval ('10min', '15min', '20min', '25min', '30min', 
                         '45min', '1hour', '2hour', '3hour', '5hour', '1day', '2day', '5day', 
                         '1week', '2week', '1month', '2month', '3month', '5month', '1year')
        
        Returns:
        - bool: True if successful, False otherwise
        """
        if self.original_data is None:
            print("No data loaded. Please load data first.")
            return False
        
        try:
            # Map user-friendly intervals to pandas resample rule
            interval_map = {
                '5min': '5T', '10min': '10T', '15min': '15T', '20min': '20T', 
                '25min': '25T', '30min': '30T', '45min': '45T',
                '1hour': 'H', '2hour': '2H', '3hour': '3H', '5hour': '5H',
                '1day': 'D', '2day': '2D', '5day': '5D',
                '1week': 'W', '2week': '2W',
                '1month': 'M', '2month': '2M', '3month': '3M', '5month': '5M',
                '1year': 'Y'
            }
            
            if interval not in interval_map:
                print(f"Invalid interval: {interval}. Please use one of the supported intervals.")
                return False
            
            # Use the current filtered data if filtered by date, otherwise use original data
            if self.filtered_by_date:
                data = self.data.copy()
            else:
                data = self.original_data.copy()
                
            # Resample the data
            resampled = data.resample(interval_map[interval])
            
            # Aggregate using common OHLCV logic
            resampled_data = pd.DataFrame({
                'open': resampled['open'].first(),
                'high': resampled['high'].max(),
                'low': resampled['low'].min(),
                'close': resampled['close'].last(),
                'volume': resampled['volume'].sum()
            })
            
            # Update the current data with resampled data
            self.data = resampled_data
            self.current_interval = interval
            
            print(f"Data resampled to {interval} interval with {len(resampled_data)} rows")
            return True
            
        except Exception as e:
            print(f"Error resampling data: {e}")
            return False
    
    def analyze_price_movements(self, percentage):
        """
        Analyze how many times the price went up or down from the open price by a specified percentage
        
        Parameters:
        - percentage (float): The percentage threshold for price movements
        
        Returns:
        - dict: Dictionary containing analysis results
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        try:
            # Convert percentage to decimal form
            percent_decimal = percentage / 100
            
            # Calculate the up and down thresholds based on open price
            self.data['up_threshold'] = self.data['open'] * (1 + percent_decimal)
            self.data['down_threshold'] = self.data['open'] * (1 - percent_decimal)
            
            # Movement categories
            # 1. Price went up by at least the specified percentage (high >= up_threshold)
            up_moves = self.data[self.data['high'] >= self.data['up_threshold']]
            
            # 2. Price went down by at least the specified percentage (low <= down_threshold)
            down_moves = self.data[self.data['low'] <= self.data['down_threshold']]
            
            # 3. Price both went up and down by the specified percentage in the same candle
            both_moves = self.data[(self.data['high'] >= self.data['up_threshold']) & 
                                   (self.data['low'] <= self.data['down_threshold'])]
            
            # 4. Price closed higher than open (bullish candle)
            bullish_candles = self.data[self.data['close'] > self.data['open']]
            
            # 5. Price closed lower than open (bearish candle)
            bearish_candles = self.data[self.data['close'] < self.data['open']]
            
            # 6. Price went up by the percentage and closed higher than open
            up_and_bullish = self.data[(self.data['high'] >= self.data['up_threshold']) & 
                                       (self.data['close'] > self.data['open'])]
            
            # 7. Price went down by the percentage and closed lower than open
            down_and_bearish = self.data[(self.data['low'] <= self.data['down_threshold']) & 
                                        (self.data['close'] < self.data['open'])]
            
            # 8. Price went up by the percentage but closed lower than open (failed rally)
            up_but_bearish = self.data[(self.data['high'] >= self.data['up_threshold']) & 
                                      (self.data['close'] < self.data['open'])]
            
            # 9. Price went down by the percentage but closed higher than open (failed drop)
            down_but_bullish = self.data[(self.data['low'] <= self.data['down_threshold']) & 
                                        (self.data['close'] > self.data['open'])]
            
            # Store the results in a dictionary
            results = {
                'total_candles': len(self.data),
                'current_interval': self.current_interval,
                'percentage_threshold': percentage,
                'up_moves': len(up_moves),
                'down_moves': len(down_moves),
                'both_up_and_down': len(both_moves),
                'bullish_candles': len(bullish_candles),
                'bearish_candles': len(bearish_candles),
                'up_and_bullish': len(up_and_bullish),
                'down_and_bearish': len(down_and_bearish),
                'up_but_bearish': len(up_but_bearish),
                'down_but_bullish': len(down_but_bullish),
                'up_moves_percentage': round(len(up_moves) / len(self.data) * 100, 2) if len(self.data) > 0 else 0,
                'down_moves_percentage': round(len(down_moves) / len(self.data) * 100, 2) if len(self.data) > 0 else 0,
                'bullish_percentage': round(len(bullish_candles) / len(self.data) * 100, 2) if len(self.data) > 0 else 0,
                'bearish_percentage': round(len(bearish_candles) / len(self.data) * 100, 2) if len(self.data) > 0 else 0
            }
            
            return results
            
        except Exception as e:
            print(f"Error analyzing price movements: {e}")
            return None
    
    def plot_price_movement_analysis(self, percentage, show_plot=True, save_path=None):
        """
        Plot the price movement analysis results
        
        Parameters:
        - percentage (float): The percentage threshold that was used in the analysis
        - show_plot (bool): Whether to display the plot
        - save_path (str, optional): Path to save the plot
        
        Returns:
        - None
        """
        results = self.analyze_price_movements(percentage)
        
        if not results:
            return
        
        # Create a figure and axis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Data for the first pie chart (Price Movements)
        movement_labels = ['Up Moves', 'Down Moves', 'Both Up & Down', 'Neither']
        movement_sizes = [
            results['up_moves'] - results['both_up_and_down'],
            results['down_moves'] - results['both_up_and_down'],
            results['both_up_and_down'],
            results['total_candles'] - results['up_moves'] - results['down_moves'] + results['both_up_and_down']
        ]
        movement_colors = ['green', 'red', 'orange', 'gray']
        
        # Plot the first pie chart
        ax1.pie(movement_sizes, labels=movement_labels, colors=movement_colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.set_title(f'Price Movements (±{percentage}%)')
        
        # Data for the second pie chart (Candle Types)
        candle_labels = ['Bullish', 'Bearish', 'Doji']
        candle_sizes = [
            results['bullish_candles'],
            results['bearish_candles'],
            results['total_candles'] - results['bullish_candles'] - results['bearish_candles']
        ]
        candle_colors = ['green', 'red', 'blue']
        
        # Plot the second pie chart
        ax2.pie(candle_sizes, labels=candle_labels, colors=candle_colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax2.set_title('Candle Types')
        
        # Add a title to the figure
        fig.suptitle(f'Price Movement Analysis ({self.current_interval} interval)', fontsize=16)
        
        # Add a text box with detailed statistics
        textstr = '\n'.join((
            f'Total Candles: {results["total_candles"]}',
            f'Percentage Threshold: {percentage}%',
            f'Up Moves: {results["up_moves"]} ({results["up_moves_percentage"]}%)',
            f'Down Moves: {results["down_moves"]} ({results["down_moves_percentage"]}%)',
            f'Both Up & Down: {results["both_up_and_down"]}',
            f'Up & Bullish: {results["up_and_bullish"]}',
            f'Down & Bearish: {results["down_and_bearish"]}',
            f'Up but Bearish: {results["up_but_bearish"]}',
            f'Down but Bullish: {results["down_but_bullish"]}'
        ))
        
        fig.text(0.5, 0.02, textstr, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.9])
        
        # Save the plot if a save path is provided
        if save_path:
            plt.savefig(save_path)
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()


def main():
    """
    Main function demonstrating the usage of StockAnalyzer
    """
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Analyze intraday stock data')
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--date', '-d', type=str, help='Analyze only specific date (YYYY-MM-DD)')
    parser.add_argument('--interval', '-i', type=str, default='5min', 
                        help='Time interval for analysis (e.g., 5min, 15min, 1hour)')
    parser.add_argument('--percentage', '-p', type=float, default=1.0,
                        help='Percentage threshold for price movement analysis')
    parser.add_argument('--output', '-o', type=str, default='price_movement_analysis.png',
                        help='Output file path for the visualization')
    
    args = parser.parse_args()
    
    print("Intraday Stock Analyzer for Gold")
    print("--------------------------------")
    
    # Initialize the analyzer
    analyzer = StockAnalyzer()
    
    # Load the data
    csv_file_path = args.file
    print(f"Attempting to load data from {csv_file_path}...")
    success = analyzer.load_data(csv_file_path)
    
    if not success:
        print("Failed to load data. Please check the file path and format.")
        return
    
    # Filter by date if specified
    if args.date:
        print(f"\nFiltering data for date: {args.date}")
        if not analyzer.filter_by_date(args.date):
            print("Failed to filter by date. Please check the date format and ensure data exists for that date.")
            return
    
    # Resample to the specified interval
    print(f"\nResampling data to {args.interval} intervals...")
    analyzer.resample_data(args.interval)
    
    # Analyze price movements with the specified threshold
    percentage = args.percentage
    print(f"\nAnalyzing price movements with {percentage}% threshold...")
    results = analyzer.analyze_price_movements(percentage)
    
    if results:
        print("\nAnalysis Results:")
        print(f"Total candles: {results['total_candles']}")
        print(f"Up moves: {results['up_moves']} ({results['up_moves_percentage']}%)")
        print(f"Down moves: {results['down_moves']} ({results['down_moves_percentage']}%)")
        print(f"Both up & down: {results['both_up_and_down']}")
        print(f"Bullish candles: {results['bullish_candles']} ({results['bullish_percentage']}%)")
        print(f"Bearish candles: {results['bearish_candles']} ({results['bearish_percentage']}%)")
        print(f"Up & Bullish: {results['up_and_bullish']}")
        print(f"Down & Bearish: {results['down_and_bearish']}")
        print(f"Up but Bearish (failed rally): {results['up_but_bearish']}")
        print(f"Down but Bullish (failed drop): {results['down_but_bullish']}")
        
        # Plot the results
        print("\nGenerating visualization...")
        analyzer.plot_price_movement_analysis(percentage, save_path=args.output)
        print(f"Visualization saved as '{args.output}'")


if __name__ == "__main__":
    main()