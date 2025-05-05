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
            
            # Make a copy of the original data
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


# Example of how to use the StockAnalyzer class
def main():
    """
    Main function demonstrating the usage of StockAnalyzer
    """
    # Replace this with your actual CSV file path
    csv_file_path = './XAU_5m_data_2004_to_2024-09-20.csv'
    
    print("Intraday Stock Analyzer for Gold")
    print("--------------------------------")
    
    # Initialize the analyzer
    analyzer = StockAnalyzer()
    
    # Load the data
    print(f"Attempting to load data from {csv_file_path}...")
    success = analyzer.load_data(csv_file_path)
    
    if not success:
        print("Failed to load data. Please check the file path and format.")
        return
    
    # Example: Resample to 15-minute intervals
    print("\nResampling data to 15-minute intervals...")
    analyzer.resample_data('15min')
    
    # Example: Analyze price movements with a 1% threshold
    percentage = 1.0
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
        
        # Plot the results
        print("\nGenerating visualization...")
        analyzer.plot_price_movement_analysis(percentage, save_path='price_movement_analysis.png')
        print("Visualization saved as 'price_movement_analysis.png'")


if __name__ == "__main__":
    main()