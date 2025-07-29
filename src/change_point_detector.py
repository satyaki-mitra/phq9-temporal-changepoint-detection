# DEPENDENCIES
import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Optional
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from scipy.stats import ttest_ind


# IGNORE ALL FUTURE WARNINGS DURING RUN TIME
warnings.filterwarnings('ignore')


class ChangePointDetector:
    """
    A comprehensive class for detecting change points in PHQ-9 score data using PELT algorithm,
    validating the detected change points through statistical testing, and visualizing the results
    """
    def __init__(self, data_path: str, penalty: float, jump: int, minimum_points: int, smoothing_window_size: int, alpha: float,
                 final_plot_path: str, aggregated_plot_path: str, validated_plot_path: str) -> None:
        """
        Initialize the ChangePointDetector with configuration parameters
        
        Arguments:
        ----------
            data_path { str } : Path to the CSV file containing PHQ-9 data
            
            penalty              { float } : Penalty parameter for PELT algorithm 
            
            jump                  { int }  : Jump parameter for subsampling 
            
            minimum_points        { int }  : Minimum number of points for cluster formation
            
            smoothing_window_size { int }  : Window size for moving average calculation
            
            alpha                { float } : Significance level for statistical testing
            
            final_plot_path       { str }  : Path to save final scatter plot with segments
            
            aggregated_plot_path  { str }  : Path to save aggregated data plot
            
            validated_plot_path   { str }  : Path to save validation plot
        """
        self.data_path             = data_path
        self.penalty               = penalty
        self.jump                  = jump
        self.minimum_points        = minimum_points
        self.smoothing_window_size = smoothing_window_size
        self.alpha                 = alpha
        self.final_plot_path       = final_plot_path
        self.aggregated_plot_path  = aggregated_plot_path
        self.validated_plot_path   = validated_plot_path
        

    def load_data(self) -> pd.DataFrame:
        """
        Load PHQ-9 data from the specified CSV file
        
        Returns:
        --------
            { pd.DataFrame }  : Loaded PHQ-9 data
            
        Raises:
        -------
            FileNotFoundError : If the data file doesn't exist

            Exception         : For any other loading errors
        """
        try:
            raw_data            = pd.read_csv(filepath_or_buffer = self.data_path, 
                                              index_col          = 'Day',
                                             )

            transposed_raw_data = raw_data.T

            return transposed_raw_data
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        
        except Exception as e:
            raise Exception(f"Error loading data: {repr(e)}")

    
    def calculate_aggregated_statistic(self, input_data: pd.DataFrame) -> pd.Series:
        """
        Calculate the coefficient of variation (CV) for each day's PHQ-9 scores across all patients
        
        Arguments:
        ----------
            input_data { pd.DataFrame } : A pandas DataFrame containing raw dataset of PHQ-9 scores of all sample
                                          patients for all sample days
        
        Returns:
        --------
                  { pd.Series }         : Series containing coefficient of variation values for each day
            
        Raises:
        -------
            TypeError                   : If input_data is not a pandas DataFrame
            
            Exception                   : For any calculation errors
        """
        # Checking the pre-defined data types of the input arguments
        if (not isinstance(input_data, pd.DataFrame)):
            raise TypeError(f'Expected a pandas DataFrame for the argument: input_data, got: {type(input_data)} instead')
        
        try:
            # Calculate the coefficient of variation (CV) for each day's scores (across all patients)
            transposed_data      = input_data.T
            mean_values          = transposed_data.apply(lambda x: np.nanmean(x), axis = 1)
            std_values           = transposed_data.apply(lambda y: np.nanstd(y), axis = 1)
            aggregated_data      = (std_values / mean_values)

            return aggregated_data
        
        except Exception as e:
            raise Exception(f'DataAggregationError: Got: {repr(e)} while calculating aggregated statistic from the raw dataset')

        
    def detect_change_points(self, raw_data: pd.DataFrame, aggregated_data: pd.Series) -> Tuple[List[pd.DataFrame], List[int], Dict]:
        """
        Detect change points in PHQ-9 scores using PELT algorithm and segregate data accordingly
        
        Arguments:
        ----------
            raw_data      { pd.DataFrame } : A pandas DataFrame containing raw dataset of PHQ-9 scores of all sample patients for all sample days
            
            aggregated_data { pd.Series }   : A pandas Series containing the aggregated values of PHQ-9 scores, where patient ID is the index of the series
        
        Returns:
        --------
                      { tuple }             : A tuple containing: - sections: List of DataFrames representing segments of the input data
                                                                  - shift_points: List of the break points / change points indices of the dataset
                                                                  - cluster_boundaries: Dictionary containing cluster boundaries information
                
        Raises:
        -------
            TypeError                       : If input arguments are not of their pre-defined data types
            
            Exception                       : For any errors during change point detection
        """
        # Checking the pre-defined data types of the input arguments
        if (not isinstance(raw_data, pd.DataFrame)):
            raise TypeError(f'Expected a pandas DataFrame for the argument: raw_data, got: {type(raw_data)} instead')

        if (not isinstance(aggregated_data, pd.Series)):
            raise TypeError(f'Expected a pandas Series for the argument: aggregated_data, got: {type(aggregated_data)} instead')
        
        try:
            # Perform change point detection on the CV values : L1 Regularization / Lasso Regularization Model has been used here
            model              = "l1"                                 

            # The cost function, by minimizing which change points would be detected
            cost_function      = rpt.costs.CostL1                      
            
            algorithm          = rpt.Pelt(model       = model, 
                                          custom_cost = cost_function,
                                          min_size    = self.minimum_points,
                                          jump        = self.jump,
                                         )
            
            algorithm.fit(aggregated_data.values)
            
            # pen / penalty is a hyperparameter here
            shift_points       = algorithm.predict(pen = self.penalty)  

            # Segregate the time axis based on the change points
            sections           = list()
            cluster_boundaries = dict()
            prev_shift_point   = 0
            
            for i, shift_point in enumerate(shift_points):
                section          = raw_data.iloc[:, prev_shift_point : shift_point]
                sections.append(section)
                
                cluster_boundaries.update({f'Cluster-{i+1}': {'Start' : section.T.index[0],
                                                              'End'   : section.T.index[-1],
                                                             }
                                         })

                prev_shift_point = shift_point
                
            # Add the last section from the last change point to the end
            section = raw_data.iloc[:, prev_shift_point:]
            if (len(section.T) != 0):
                sections.append(section)
            
            else:
                pass
            
            return sections, shift_points, cluster_boundaries
            
        except Exception as e:
            raise Exception(f'SegregationError: Got: {repr(e)} while segregating PHQ-9 scores by change points')


    def test_change_point_significance(self, aggregated_data: pd.Series, change_points: list) -> dict:
        """
        Perform hypothesis testing for change point significance using statistical tests
        
        Arguments:
        ----------
            aggregated_data  { pd.Series } : A pandas Series containing the Coefficient of Variation (CV) values
            
            change_points       { list }   : A list containing all the change points detected by the detection model
        
        Returns:
        --------
                      { dict }             : Dictionary containing test results, interpretations and statistical information
            
        Raises:
        -------
            TypeError                      : If input arguments are not of expected types
            
            Exception                      : For any errors during significance testing
        """
        # Input type checking
        if (not isinstance(aggregated_data, pd.Series)):
            raise TypeError(f'Expected a pandas Series for the argument: aggregated_data, got: {type(aggregated_data)} instead')
            
        if (not isinstance(change_points, list)):
            raise TypeError(f'Expected a list for the argument: change_points, got: {type(change_points)} instead')
       
        try: 
            # Save the testing results and their interpretations in a dictionary
            test_results        = dict()
            
            # Define a list containing all the boundary points in the data after clustering
            all_boundaries_list = [0] + change_points

            # Perform the validity of Change Points iteratively
            for index in range(len(all_boundaries_list)-2):
                change_boundary_1 = (all_boundaries_list[index], all_boundaries_list[index+1])
                change_boundary_2 = (all_boundaries_list[index+1], all_boundaries_list[index+2])
                
                # Split the data into two groups based on their corresponding cluster boundaries
                group_1_data      = aggregated_data[change_boundary_1[0]:change_boundary_1[1]]
                group_2_data      = aggregated_data[change_boundary_2[0]:change_boundary_2[1]]
        
                # Extract only the values of both the groups in numpy 1-d arrays
                group_1_values    = group_1_data.values
                group_2_values    = group_2_data.values
            
                # Perform a statistical test (Wilcoxon rank-sum test)
                if ((len(np.unique(group_1_values)) == 1) or (len(np.unique(group_2_values)) == 1)):
                    # If either group has constant values, use t-test
                    test_statistic, p_value = ttest_ind(group_1_data, group_2_data)
                    test_name               = 'T-Test'
                
                else:
                    test_statistic, p_value = ranksums(group_1_data, group_2_data)
                    test_name               = 'Wilcoxon Rank-Sum Test'

                # Make the interpretation on the basis of p-value of the test, by comparing with alpha
                if (p_value < self.alpha):
                    interpretation = f"The change point is statistically significant (p-value = {p_value:.4f})"
                
                else:
                    interpretation = f"The change point is not statistically significant (p-value = {p_value:.4f})"
                
                test_results.update({(change_boundary_1, change_boundary_2): {'test_name'            : test_name,
                                                                              'p-value'              : p_value,
                                                                              'test_statistic_value' : test_statistic,
                                                                              'interpretation'       : interpretation,
                                                                             }
                                   })

            return test_results
        
        except Exception as e:
            raise Exception(f'SignificanceTestingError: {repr(e)}')

    
    def visualize_aggregated_data(self, aggregated_data: pd.Series, plot_location: str = None) -> None:
        """
        Visualize aggregated PHQ-9 data as a line chart
        
        Arguments:
        ----------
            aggregated_data { pd.Series } : A pandas Series containing aggregated PHQ-9 data across all study patients
            
            plot_location      { str }    : Location to save the plot (if None, plot is displayed)
                
        Raises:
        -------
            TypeError                     : If input arguments are not of specified data types
            
            Exception                     : For any plotting errors
        """
        # Checking the pre-defined data types of the input arguments
        if not isinstance(aggregated_data, pd.Series):
            raise TypeError(f'Expected a pandas Series for the argument: aggregated_data, got: {type(aggregated_data)} instead')
        
        if plot_location is not None and not isinstance(plot_location, str):
            raise TypeError(f'Expected a string for the argument: plot_location, got: {type(plot_location)} instead')

        try:
            # The aggregated data for each day across all the patients
            y_axis_values = aggregated_data.values
            x_axis_values = aggregated_data.index
            
            # Set the plot size
            plt.figure(figsize = (15, 10))
            
            # Plotting in line chart
            plt.plot(x_axis_values, 
                     y_axis_values, 
                     label  = 'Coefficient of Variation (CV)', 
                     c      = 'green', 
                     marker = 'o',
                    )
            
            # Setting labels of X-Y axis & Title 
            plt.xlabel(xlabel   = 'Time Axis (Day)', 
                       fontsize = 15,
                      )

            plt.ylabel(ylabel   = 'Daily Coefficient of Variation of PHQ-9 Score',
                       fontsize = 15,
                      )

            plt.xticks(ticks    = np.arange(0, len(x_axis_values), 1),
                       labels   = x_axis_values,
                       rotation = 90,
                      )

            plt.yticks(ticks    = np.arange(0, np.ceil(max(y_axis_values))+1, 0.25),
                       labels   = np.arange(0, np.ceil(max(y_axis_values))+1, 0.25),
                       rotation = 0,
                      )

            plt.title(label    = 'Coefficient of Variation of PHQ-9 Score Across All Patients Over Time\n',
                      fontsize = 20,
                     )
            
            # Decorators
            plt.legend(bbox_to_anchor = (1.0, 0.5),
                       loc            = 'upper left',
                       prop           = {'size': 10},
                      )

            plt.grid(True)
            
            # Output
            if plot_location is not None:
                plt.savefig(fname       = plot_location, 
                            bbox_inches = "tight",
                           )
            
            else:
                plt.show()

        except Exception as e:
            raise Exception(f'AggregatedDataPlottingError: Got: {repr(e)} while plotting the aggregated data')
    

    def visualize_segregated_data(self, input_data: pd.DataFrame, aggregated_data: pd.Series, change_points: list, plot_path: str = None) -> None:
        """
        Visualize segregated data with scatter points, trend line, and change points
        
        Arguments:
        ----------
            input_data    { pd.DataFrame } : Raw PHQ-9 scores dataset
            
            aggregated_data { pd.Series }  : Aggregated PHQ-9 values

            change_points      { list }    : List of change point indices
            
            plot_path           { str }    : Path to save the plot
                
        Raises:
        -------
            TypeError                      : If input arguments are not of expected types
            
            Exception                      : For any plotting errors
        """
        # Type checking
        if (not isinstance(input_data, pd.DataFrame)):
            raise TypeError(f'Expected a pandas DataFrame for input_data, got: {type(input_data)}')
        
        if (not isinstance(aggregated_data, pd.Series)):
            raise TypeError(f'Expected a pandas Series for aggregated_data, got: {type(aggregated_data)}')
        
        if (not isinstance(change_points, list)):
            raise TypeError(f'Expected a list for change_points, got: {type(change_points)}')
        
        if (plot_path is not None and (not isinstance(plot_path, str))):
            raise TypeError(f'Expected a string for plot_path, got: {type(plot_path)}')
        
        try:
            # Prepare data for visualization
            num_days               = len(input_data.columns)
            x_axis_values          = np.arange(num_days)  # Time axis / X-Axis

            # Daily Average PHQ-9 Scores of observed patients
            daily_average          = input_data.mean(axis   = 0, 
                                                     skipna = True,
                                                    )
            
            # Calculate Moving Average with defined window size
            smoothed_daily_average = daily_average.rolling(window      = self.smoothing_window_size,
                                                           min_periods = self.smoothing_window_size,
                                                           center      = True,
                                                           win_type    = None, 
                                                           axis        = 0,
                                                           closed      = 'both',
                                                           method      = 'single',
                                                          ).mean()
             
            # Visualize the segments with proper boundaries
            fig, ax                = plt.subplots(figsize = (20, 12))

            # Plot individual patient scores as scatter points
            # Flatten the data to get all individual scores with their corresponding day indices
            for day_idx in range(num_days):
                day_scores = input_data.iloc[:, day_idx].dropna()  # Get all patient scores for this day
                day_x      = np.full(len(day_scores), day_idx)     # X-coordinates for this day
                
                ax.scatter(day_x, 
                           day_scores, 
                           color = 'brown', 
                           alpha = 0.7,
                           s     = 30,
                          )
            
            # Plot the smoothed trend line
            ax.plot(x_axis_values,
                    smoothed_daily_average,
                    scalex     = True, 
                    scaley     = True,
                    color      = 'red',
                    marker     = '^',
                    markersize = 8.0,
                    linestyle  = '-',
                    linewidth  = 2.0, 
                    label      = 'Smoothed Daily Average',
                   )

            # Highlight the segments as vertical spans with different colors
            boundaries   = np.concatenate([[0], change_points, [num_days]])
            num_segments = len(boundaries) - 2

            # Create color map
            colors       = plt.cm.get_cmap('tab20', num_segments)
            
            for i in range(num_segments):
                start_boundary = boundaries[i]
                end_boundary   = boundaries[i+1]
                
                ax.axvspan(start_boundary, 
                           end_boundary, 
                           facecolor = colors(i), 
                           alpha     = 0.35,
                           label     = f'Section - {i+1}',
                          )

            # Set labels and title
            ax.set_xlabel(xlabel   = 'Survey Day (Time)', 
                          fontsize = 15,
                         )

            ax.set_ylabel(ylabel   = 'Daily PHQ-9 Scores', 
                          fontsize = 15,
                         )

            ax.set_title(label    = 'Scatter Plot of PHQ-9 scores With Corresponding Segments According To Underlying Pattern\n', 
                         fontsize = 25,
                        )

            # Set axis ticks
            plt.xticks(ticks    = range(num_days), 
                       labels   = input_data.columns,
                       rotation = 90,
                      )

            plt.yticks(ticks    = range(0, 29), 
                       labels   = range(0, 29),
                       rotation = 0,
                      )

            # Layout adjustments
            plt.tight_layout(pad   = 2.0,
                             h_pad = 1.0, 
                             w_pad = 1.0,
                            )

            ax.legend(bbox_to_anchor = (1.00, 1.00),
                      loc            = 'upper left',
                      prop           = {'size': 12})
            
            # Output
            if plot_path is not None:
                plt.savefig(fname       = plot_path,
                            bbox_inches = 'tight',
                           )
            else:
                plt.show()
            
        except Exception as e:
            raise Exception(f'SegregatedDataPlottingError: Got: {repr(e)} while plotting segregated data')


    def visualize_aggregated_data_with_clusters(self, aggregated_data: pd.Series, change_points: list, plot_location: str = None) -> None:
        """
        Visualize aggregated PHQ-9 data with change points marked as vertical lines
        
        Arguments:
        ----------
            aggregated_data { pd.Series } : Aggregated PHQ-9 data across all patients
            
            change_points      { list }   : List of detected change points
            
            plot_location       { str }   : Location to save the plot
                
        Raises:
        -------
            TypeError                     : If input arguments are not of expected types
            
            Exception                     : For any plotting errors
        """
        # Type checking
        if (not isinstance(aggregated_data, pd.Series)):
            raise TypeError(f'Expected a pandas Series for aggregated_data, got: {type(aggregated_data)}')
        
        if (not isinstance(change_points, list)):
            raise TypeError(f'Expected a list for change_points, got: {type(change_points)}')
        
        if (plot_location is not None and (not isinstance(plot_location, str))):
            raise TypeError(f'Expected a string for plot_location, got: {type(plot_location)}')

        try:
            # The aggregated data for each day across all the patients
            y_axis_values = aggregated_data.values
            x_axis_values = aggregated_data.index
            
            # Set the plot size
            plt.figure(figsize = (15, 7))
            
            # Plotting in line chart
            plt.plot(x_axis_values, 
                     y_axis_values, 
                     label  = 'Coefficient of Variation (CV)', 
                     c      = 'purple', 
                     marker = '*',
                    )
            
            # Highlight the detected change points with red vertical lines
            for i, change_point in enumerate(change_points[:-1]):
                plt.axvline(x         = x_axis_values[change_point-1], 
                            color     = 'red', 
                            linestyle = '--',
                            alpha     = 1, 
                            label     = f'Change Point - {i+1}',
                           )

            # Add value annotations
            for index, aggregate in enumerate(aggregated_data):
                plt.text(x        = index, 
                         y        = aggregate+0.15,
                         s        = f'{aggregate:.3f}',
                         fontsize = 10,
                         ha       = 'center',
                         va       = 'bottom',
                         color    = 'black', 
                         rotation = 90,
                        )   
            
            # Setting labels for X-Y axis & Title
            plt.xlabel(xlabel   = 'Time Axis (Day)', 
                       fontsize = 15,
                      )

            plt.ylabel(ylabel   = 'Daily Coefficient of Variation of PHQ-9 Score', 
                       fontsize = 15,
                      )

            plt.xticks(ticks    = np.arange(0, len(x_axis_values), 1),
                       labels   = x_axis_values,
                       rotation = 90,
                      )

            plt.yticks(ticks    = np.arange(0, np.ceil(max(y_axis_values))+1, 0.25),
                       labels   = np.arange(0, np.ceil(max(y_axis_values))+1, 0.25),
                       rotation = 0,
                      )

            plt.title(label    = 'Coefficient of Variation of PHQ-9 Score Across All Patients Over Time\n',
                      fontsize = 20,
                     )
            
            # Decorators
            plt.legend(bbox_to_anchor = (1.0, 0.5),
                       loc            = 'upper left',
                       prop           = {'size': 10},
                      )

            plt.grid(True)
            
            # Output
            if plot_location is not None:
                plt.savefig(fname       = plot_location, 
                            bbox_inches = "tight",
                           )
            
            else:
                plt.show()

        except Exception as e:
            raise Exception(f'AggregatedDataClusterPlottingError: Got: {repr(e)} while plotting aggregated data with clusters')


    def run_full_analysis(self) -> dict:
        """
        Run the complete change point detection analysis pipeline
        
        Returns:
        --------
            { dict }  : Dictionary containing all analysis results
            
        Raises:
        -------
            Exception : For any errors during the analysis pipeline
        """
        try:
            # Load data
            print("Loading data...")
            raw_data                                    = self.load_data()
            
            # Calculate aggregated statistics
            print("Calculating aggregated statistics...")
            aggregated_data                             = self.calculate_aggregated_statistic(input_data = raw_data)
            
            # Detect change points
            print("Detecting change points...")
            sections, change_points, cluster_boundaries = self.detect_change_points(raw_data        = raw_data, 
                                                                                    aggregated_data = aggregated_data,
                                                                                   )
            
            # Test significance of change points
            print("Testing change point significance...")
            test_results                                = self.test_change_point_significance(aggregated_data = aggregated_data, 
                                                                                              change_points   = change_points,
                                                                                             )
            
            # Generate visualizations
            print("Generating visualizations...")
            self.visualize_aggregated_data(aggregated_data = aggregated_data, 
                                           plot_location   = self.aggregated_plot_path,
                                          )

            self.visualize_aggregated_data_with_clusters(aggregated_data = aggregated_data, 
                                                         change_points   = change_points, 
                                                         plot_location   = self.validated_plot_path,
                                                        )

            self.visualize_segregated_data(input_data      = raw_data, 
                                           aggregated_data = aggregated_data, 
                                           change_points   = change_points,
                                           plot_path       = self.final_plot_path,
                                          )
            
            # Compile results
            results = {'raw_data_shape'           : raw_data.shape,
                       'num_change_points'        : len(change_points) - 1,  # Subtract 1 because last point is end of data
                       'change_points'            : change_points,
                       'cluster_boundaries'       : cluster_boundaries,
                       'statistical_test_results' : test_results,
                       'aggregated_statistics'    : aggregated_data,
                      }
            
            print("Analysis completed successfully!")
            
            return results
            
        except Exception as e:
            raise Exception(f"Error during full analysis: {repr(e)}")