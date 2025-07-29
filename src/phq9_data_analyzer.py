# DEPENDENCIES
import json
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


# IGNORE WARNINGS
warnings.filterwarnings('ignore')

# SET PLOTTING STYLES
plt.style.use('default')
sns.set_palette("husl")


class PHQ9DataAnalyzer:
    """
    A class for loading, analyzing, visualizing and clustering PHQ-9 data to understand temporal patterns and patient groups
    """
    def __init__(self, data_path: str = None, figsize: tuple = (15, 12)):
        """
        Initialize the PHQ-9 data analyzer
        
        Arguments:
        ----------
            data_path  { str }  : Path to the PHQ-9 data CSV file

            figsize   { tuple } : Default figure size for plots
        """
        self.logger         = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.figsize        = figsize
        self.data           = None
        self.processed_data = None
        
        if data_path:
            self.load_data(data_path)
        
        self.logger.info("Initialized PHQ9DataAnalyzer")

    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load PHQ-9 data from CSV file
        
        Arguments:
        ----------
            data_path { str } : Path to the CSV file

        Raises:
        -------
            FileNotFoundError : If file doesn't exist

            Exception         : For other loading errors
            
        Returns:
        --------
            { pd.DataFrame }  : Loaded data
        """
        try:
            data_path = Path(data_path)

            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            self.data = pd.read_csv(filepath_or_buffer = data_path, 
                                    index_col          = 0,
                                   )
            
            # Log data statistics
            self.logger.info(f"Loaded data from: {data_path}")
            self.logger.info(f"Data shape: {self.data.shape}")
            self.logger.info(f"Missing values: {self.data.isna().sum().sum()}")
            
            return self.data
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {data_path}")
            raise

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise Exception(f"DataLoadingError: {e}")
    

    def validate_data(self, data: pd.DataFrame = None) -> bool:
        """
        Validate PHQ-9 data format and content
        
        Arguments:
        ----------
            data { pd.DataFrame } : Data to validate
            
        Returns:
        --------
                 { bool }         : True if valid, False otherwise
        """
        try:
            if data is None:
                data = self.data
            
            if data is None:
                self.logger.error("No data available for validation")
                return False
            
            if not isinstance(data, pd.DataFrame):
                self.logger.error(f"Expected DataFrame, got {type(data)}")
                return False
            
            # Check if PHQ-9 scores are in valid range (0-27)
            numeric_data = data.select_dtypes(include=[np.number])
            
            if not numeric_data.empty:
                min_score = numeric_data.min().min()
                max_score = numeric_data.max().max()
                
                if ((min_score < 0) or (max_score > 27)):
                    self.logger.warning(f"PHQ-9 scores outside valid range (0-27): {min_score}-{max_score}")
            
            self.logger.info("Data validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during data validation: {e}")
            return False
    

    def get_summary_statistics(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate summary statistics for the dataset
        
        Arguments:
        ----------
            data { pd.DataFrame } : Data to analyze (uses self.data if None)
            
        Returns:
        --------
             { pd.DataFrame }     : Summary statistics
        """
        try:
            if data is None:
                data = self.data
            
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            # Calculate summary statistics
            summary_stats                       = data.T.describe().T
            
            # Add additional statistics
            summary_stats['missing_count']      = data.isna().sum()
            summary_stats['missing_percentage'] = (data.isna().sum() / len(data)) * 100
            
            self.logger.info("Summary statistics calculated successfully")
            
            return summary_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating summary statistics: {e}")
            raise Exception(f"SummaryStatisticsError: {e}")
    

    def create_scatter_plot(self, data: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Create scatter plot of PHQ-9 scores for all query days
        
        Arguments:
        ----------
            data { pd.DataFrame } : Data to plot (uses self.data if None)
            
            save_path { str }     : Path to save the plot (optional)
        """
        try:
            if data is None:
                data = self.data
            
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            # Create figure
            fig, ax   = plt.subplots(figsize = self.figsize)
            
            # Extract and format data
            data_dict = data.T.to_dict()
            keys      = list(data_dict.keys())
            values    = []
            
            for key in keys:
                scores = [x for x in list(data_dict[key].values()) if not pd.isna(x)]
                values.append(scores)
            
            # Calculate circle sizes based on number of values
            circle_sizes = [len(value) for value in values]
            
            # Plot PHQ-9 scores for each day
            for key, value, size in zip(keys, values, circle_sizes):

                # Only plot if there are values
                if (size > 0):  
                    color = np.linspace(start = 0, 
                                        stop  = 1, 
                                        num   = size,
                                       )

                    ax.scatter(x     = [key]*size, 
                               y     = [element for element in value],
                               c     = color,
                               cmap  = 'viridis',
                               s     = 25,
                               alpha = 0.7,
                              )
            
            # Set color scale
            sm   = plt.cm.ScalarMappable(cmap = 'viridis')
            cbar = plt.colorbar(mappable = sm, 
                                ax       = ax,
                               )
            cbar.set_label(label    = 'Score Frequency Weights', 
                           fontsize = 12,
                          )
            
            # Set labels and formatting
            plt.xticks(ticks    = range(len(keys)), 
                       labels   = keys, 
                       fontsize = 10, 
                       rotation = 90,
                      )

            plt.yticks(ticks    = np.arange(0, 28, 2), 
                       fontsize = 10,
                      )

            plt.legend()
            plt.grid(True, alpha = 0.3)
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(fname       = save_path, 
                            dpi         = 300, 
                            bbox_inches = 'tight',
                           )

                self.logger.info(f"Daily average plot saved to: {save_path}")
            
            plt.show()
            
            # Calculate and log daily average statistics (FIX: Define daily_average_scores)
            daily_average_scores = data.mean(axis=1, skipna=True)
            self.logger.info(f"Daily average range: {daily_average_scores.min():.2f} - {daily_average_scores.max():.2f}")
            
        except Exception as e:
            self.logger.error(f"Error creating daily average plot: {e}")
            raise Exception(f"DailyAveragePlotError: {e}")
    

    def find_optimal_clusters(self, data: pd.DataFrame = None, max_clusters: int = 20, save_plots: bool = False, plot_dir: str = None) -> tuple:
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        
        Args:
            data        { pd.DataFrame } : Data for clustering (uses self.data if None)

            max_clusters     { int }     : Maximum number of clusters to test

            save_plots       { bool }    : Whether to save plots
            
            plot_dir         { str }     : Directory to save plots
            
        Returns:
        --------
                    { tuple }            : A tuple containing (elbow_point, optimal_clusters_silhouette)
        """
        try:
            if data is None:
                data = self.data
            
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            # Prepare data for clustering
            required_dataframe = data.T.fillna(value = -1,
                                               axis  = 0,
                                              )
            
            # Initialize lists for results
            inertia_list       = list()
            silhouette_scores  = list()
            
            self.logger.info(f"Testing clustering with 2 to {max_clusters+1} clusters")
            
            # Test different numbers of clusters
            for n_clusters in range(2, max_clusters + 2):
                # Fit K-means
                kmeans           = cluster.KMeans(n_clusters   = n_clusters, 
                                                  random_state = 1234,
                                                  n_init       = 10,
                                                 )

                kmeans.fit(required_dataframe)
                
                # Calculate metrics
                inertia          = kmeans.inertia_
                silhouette_score = metrics.silhouette_score(X      = required_dataframe, 
                                                            labels = kmeans.labels_,
                                                           )
                
                inertia_list.append(inertia)
                silhouette_scores.append(silhouette_score)
            
            # Find optimal clusters using elbow method
            inertia_percent_change = [100 * (inertia_list[i] - inertia_list[i-1]) / inertia_list[i-1] for i in range(1, len(inertia_list))]
            elbow_point            = inertia_percent_change.index(min(inertia_percent_change)) + 3  # +3 to account for indexing

            
            # Find optimal clusters using silhouette method
            optimal_clusters      = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 for starting from 2
            
            # Plot elbow method
            fig, (ax1, ax2)       = plt.subplots(1, 2, figsize = (20, 8))
            
            x_coordinates         = list(range(2, max_clusters + 2))
            
            # Elbow plot
            ax1.plot(x_coordinates, 
                     inertia_list, 
                     marker     = 'o',
                     markersize = 8, 
                     linewidth  = 2, 
                     color      = 'blue',
                    )

            ax1.axvline(x         = elbow_point, 
                        color     = 'red',
                        linestyle = '--', \
                        alpha     = 0.7, 
                        label     = f'Elbow Point: {elbow_point}',
                       )

            ax1.set_xlabel(xlabel   = 'Number of Clusters', 
                           fontsize = 12,
                          )

            ax1.set_ylabel(ylabel   = 'Inertia', 
                           fontsize = 12,
                          )

            ax1.set_title(label    = 'Elbow Method for Optimal Clusters', 
                          fontsize = 16,
                         )

            ax1.grid(True, alpha = 0.3)
            ax1.legend()
            
            # Silhouette plot
            ax2.plot(x_coordinates, 
                     silhouette_scores, 
                     marker     = 's', 
                     markersize = 8, 
                     linewidth  = 2, 
                     color      = 'green',
                    )

            ax2.axvline(x         = optimal_clusters, 
                        color     = 'red', 
                        linestyle = '--', 
                        alpha     = 0.7, 
                        label     = f'Optimal: {optimal_clusters}',
                       )

            ax2.set_xlabel(xlabel   = 'Number of Clusters', 
                           fontsize = 12,
                          )

            ax2.set_ylabel(ylabel   = 'Silhouette Score', 
                           fontsize = 12,
                          )

            ax2.set_title(label    = 'Silhouette Analysis for Optimal Clusters', 
                          fontsize = 16,
                         )

            ax2.grid(True, alpha = 0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plots if requested
            if save_plots and plot_dir:
                plot_path = Path(plot_dir) / "cluster_optimization.png"
                plot_path.parent.mkdir(parents  = True,
                                       exist_ok = True,
                                      )

                plt.savefig(fname       = plot_path, 
                            dpi         = 300, 
                            bbox_inches = 'tight',
                           )
                self.logger.info(f"Cluster optimization plots saved to: {plot_path}")
            
            plt.show()
            
            self.logger.info(f"Elbow point suggests {elbow_point} clusters")
            self.logger.info(f"Silhouette analysis suggests {optimal_clusters} clusters")
            
            return elbow_point, optimal_clusters
            
        except Exception as e:
            self.logger.error(f"Error in cluster optimization: {e}")
            raise Exception(f"ClusterOptimizationError: {e}")
    

    def fit_clustering_model(self, n_clusters: int, data: pd.DataFrame = None) -> np.ndarray:
        """
        Fit K-means clustering model with specified number of clusters
        
        Args:
            n_clusters { int }     : Number of clusters

            data  { pd.DataFrame } : Data for clustering (uses self.data if None)
            
        Returns:
        --------
                { np.ndarray }     : Cluster labels
        """
        try:
            if data is None:
                data = self.data
            
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            if not isinstance(n_clusters, int) or n_clusters < 2:
                raise ValueError("Number of clusters must be an integer >= 2")
            
            # Prepare data
            required_data = data.T.fillna(value=-1, axis=0)
            
            # Fit K-means
            kmeans                = cluster.KMeans(n_clusters   = n_clusters, 
                                                   random_state = 1234,
                                                   n_init       = 10,
                                                  )

            kmeans.fit(required_data.T)
            
            cluster_labels        = kmeans.labels_
            
            # Log clustering results
            unique_labels, counts = np.unique(cluster_labels, 
                                              return_counts = True,
                                             )

            self.logger.info(f"Clustering completed with {n_clusters} clusters:")
            for label, count in zip(unique_labels, counts):
                self.logger.info(f"  Cluster {label}: {count} days")
            
            return cluster_labels
            
        except Exception as e:
            self.logger.error(f"Error fitting clustering model: {e}")
            raise Exception(f"ClusterFittingError: {e}")
    

    def plot_clusters(self, cluster_labels: np.ndarray, n_clusters: int, data: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Plot clustering results with cluster boundaries
        
        Arguments:
        ----------
            cluster_labels { np.ndarray }  : Cluster labels for each day
            
            n_clusters        { int }      : Number of clusters

            data          { pd.DataFrame } : Data to plot (uses self.data if None)
            
            save_path         { str }      : Path to save the plot (optional)
        """
        try:
            if data is None:
                data = self.data
            
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            if not isinstance(cluster_labels, np.ndarray):
                raise TypeError("Cluster labels must be a numpy array")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(nrows   = 2, 
                                           ncols   = 1, 
                                           figsize = (self.figsize[0], self.figsize[1]*1.5),
                                          )

            
            # Plot 1: Scatter plot with cluster colors
            days             = list(range(len(cluster_labels)))
            colors           = plt.cm.Set3(np.linspace(0, 1, n_clusters))
            

            for day_idx, (day_name, day_data) in enumerate(data.iterrows()):
                if (day_idx < len(cluster_labels)):
                    cluster_id = cluster_labels[day_idx]
                    scores     = day_data.dropna().values
                    
                    if (len(scores) > 0):
                        ax1.scatter([day_idx] * len(scores), 
                                    scores, 
                                    c     = [colors[cluster_id]], 
                                    s     = 50, 
                                    alpha = 0.7,
                                    label = f'Cluster {cluster_id}' if day_idx == np.where(cluster_labels == cluster_id)[0][0] else "",
                                   )
            
            ax1.set_xlabel(xlabel   = 'Day Index', 
                           fontsize = 12,
                          )

            ax1.set_ylabel(ylabel   = 'PHQ-9 Score', 
                           fontsize = 12,
                          )

            ax1.set_title(label    = f'Clustering Results: {n_clusters} Clusters (Scatter View)', 
                          fontsize = 16,
                         )

            ax1.grid(True, alpha = 0.3)
            ax1.legend(bbox_to_anchor = (1.05, 1), 
                       loc            = 'upper left',
                      )
            
            # Plot 2: Daily averages with cluster boundaries
            daily_averages = data.mean(axis   = 1, 
                                       skipna = True,
                                      )
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_days = np.where(cluster_mask)[0]
                
                if (len(cluster_days) > 0):
                    ax2.scatter(cluster_days, 
                                daily_averages.iloc[cluster_days], 
                                c     = [colors[cluster_id]], 
                                s     = 100, 
                                alpha = 0.8,
                                label = f'Cluster {cluster_id}',
                               )
            
            # Add cluster boundaries
            cluster_boundaries = np.where(cluster_labels[:-1] != cluster_labels[1:])[0] + 0.5

            for boundary in cluster_boundaries:
                ax2.axvline(x         = boundary, 
                            linestyle = '--', 
                            color     = 'gray',
                            alpha     = 0.5,
                           )
            
            ax2.plot(range(len(daily_averages)), 
                     daily_averages, 
                     color     = 'black', 
                     alpha     = 0.3, 
                     linewidth = 1, 
                     label     = 'Average Trend',
                    )
            
            ax2.set_xlabel(xlabel   = 'Day Index', 
                           fontsize = 12,
                          )

            ax2.set_ylabel(ylabel   = 'Average PHQ-9 Score', 
                           fontsize = 12,
                          )

            ax2.set_title(label    = f'Daily Averages with Cluster Boundaries', 
                          fontsize = 16,
                         )

            ax2.grid(True, alpha = 0.3)
            ax2.legend(bbox_to_anchor = (1.05, 1), 
                       loc            = 'upper left',
                      )
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(fname       = save_path, 
                            dpi         = 300, 
                            bbox_inches = 'tight',
                           )

                self.logger.info(f"Cluster plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting clusters: {e}")
            raise Exception(f"ClusterPlottingError: {e}")
    

    def analyze_cluster_characteristics(self, cluster_labels: np.ndarray, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Analyze characteristics of each cluster
        
        Arguments:
        ----------
            cluster_labels { np.ndarray }  : Cluster labels

            data          { pd.DataFrame } : Data to analyze (uses self.data if None)
            
        Returns:
        --------
                { pd.DataFrame }           : Cluster characteristics
        """
        try:
            if data is None:
                data = self.data
            
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            # Calculate daily averages
            daily_averages  = data.mean(axis   = 1, 
                                        skipna = True,
                                       )
            
            # Analyze each cluster
            cluster_stats   = list()

            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                cluster_mask   = cluster_labels == cluster_id
                cluster_days   = np.where(cluster_mask)[0]
                cluster_scores = daily_averages.iloc[cluster_days]
                
                stats          = {'cluster_id'     : cluster_id,
                                  'n_days'         : len(cluster_days),
                                  'avg_score'      : cluster_scores.mean(),
                                  'std_score'      : cluster_scores.std(),
                                  'min_score'      : cluster_scores.min(),
                                  'max_score'      : cluster_scores.max(),
                                  'day_range'      : f"{cluster_days.min()}-{cluster_days.max()}",
                                  'recovery_phase' : self._classify_recovery_phase(cluster_scores.mean()),
                                 }

                cluster_stats.append(stats)
            
            cluster_df = pd.DataFrame(data = cluster_stats)
            
            self.logger.info("Cluster characteristics analyzed successfully")
            
            return cluster_df
            
        except Exception as e:
            self.logger.error(f"Error analyzing cluster characteristics: {e}")
            raise Exception(f"ClusterAnalysisError: {e}")
    

    def _classify_recovery_phase(self, avg_score: float) -> str:
        """
        Classify recovery phase based on average PHQ-9 score
        
        Arguments:
        ----------
            avg_score { float } : Average PHQ-9 score
            
        Returns:
        --------
                { str }         : Recovery phase classification
        """
        if (avg_score >= 20):
            return "Severe Depression"

        elif (avg_score >= 15):
            return "Moderately Severe Depression"

        elif (avg_score >= 10):
            return "Moderate Depression"

        elif (avg_score >= 5):
            return "Mild Depression"

        else:
            return "Minimal/No Depression"
    

    def compare_clustering_methods(self, data: pd.DataFrame = None, max_clusters: int = 10) -> pd.DataFrame:
        """
        Compare different clustering methods and their performance
        
        Arguments:
        ----------
            data    { pd.DataFrame } : Data for clustering (uses self.data if None)

            max_clusters { int }     : Maximum number of clusters to test
            
        Returns:
        --------
             { pd.DataFrame }        : Comparison results
        """
        try:
            if data is None:
                data = self.data
            
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            # Prepare data
            prepared_data      = data.T.fillna(value = -1, 
                                               axis  = 0,
                                              )
            
            comparison_results = list()
            
            # Test different numbers of clusters
            for n_clusters in range(2, max_clusters + 1):
                # K-Means
                kmeans            = cluster.KMeans(n_clusters   = n_clusters, 
                                                   random_state = 1234, 
                                                   n_init       = 10,
                                                  )

                kmeans_labels     = kmeans.fit_predict(prepared_data)

                kmeans_silhouette = metrics.silhouette_score(prepared_data, 
                                                             kmeans_labels,
                                                            )
                
                # Agglomerative Clustering
                agg               = cluster.AgglomerativeClustering(n_clusters=n_clusters)
                agg_labels        = agg.fit_predict(prepared_data)
                agg_silhouette    = metrics.silhouette_score(prepared_data, agg_labels)
                
                comparison_results.append({'n_clusters'               : n_clusters,
                                           'kmeans_silhouette'        : kmeans_silhouette,
                                           'agglomerative_silhouette' : agg_silhouette,
                                           'kmeans_inertia'           : kmeans.inertia_,
                                         })
            
            comparison_df         = pd.DataFrame(data = comparison_results)
            
            # Plot comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.plot(comparison_df['n_clusters'], 
                     comparison_df['kmeans_silhouette'], 
                     marker    = 'o', 
                     label     = 'K-Means', 
                     linewidth = 2,
                    )

            ax1.plot(comparison_df['n_clusters'], 
                     comparison_df['agglomerative_silhouette'], 
                     marker    = 's', 
                     label     = 'Agglomerative', 
                     linewidth = 2,
                    )

            ax1.set_xlabel(xlabel = 'Number of Clusters')
            ax1.set_ylabel(ylabel = 'Silhouette Score')
            ax1.set_title(label = 'Clustering Method Comparison (Silhouette Score)')
            ax1.legend()
            ax1.grid(True, alpha = 0.3)
            
            ax2.plot(comparison_df['n_clusters'], 
                     comparison_df['kmeans_inertia'], 
                     marker    = 'o', 
                     color     = 'blue', 
                     linewidth = 2,
                    )

            ax2.set_xlabel(xlabel = 'Number of Clusters')
            ax2.set_ylabel(ylabel = 'K-Means Inertia')
            ax2.set_title(label = 'K-Means Inertia vs Number of Clusters')
            ax2.grid(True, alpha = 0.3)
            
            plt.tight_layout()
            plt.show()
            
            self.logger.info("Clustering methods comparison completed")
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error comparing clustering methods: {e}")
            raise Exception(f"ClusteringComparisonError: {e}")
    

    def create_daily_average_plot(self, data: pd.DataFrame = None, save_path: str = None) -> None:
        """
        Create line plot of daily average PHQ-9 scores
        
        Arguments:
        ----------
            data { pd.DataFrame } : Data to plot (uses self.data if None)
            
            save_path { str }     : Path to save the plot (optional)
        """
        try:
            if data is None:
                data = self.data
            
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            # Calculate daily averages
            daily_averages = data.mean(axis   = 1, 
                                       skipna = True,
                                      )
            
            # Create figure
            fig, ax        = plt.subplots(figsize = self.figsize)
            
            # Plot daily averages
            days           = range(len(daily_averages))
            ax.plot(days, 
                    daily_averages, 
                    marker     = 'o', 
                    linewidth  = 2, 
                    markersize = 6, 
                    color      = 'blue',
                    alpha      = 0.7,
                   )
            
            # Add trend line
            z              = np.polyfit(days, daily_averages, 1)
            p              = np.poly1d(z)
            ax.plot(days, 
                    p(days), 
                    linestyle = '--', 
                    color     = 'red', 
                    alpha     = 0.8,
                    label     = f'Trend (slope: {z[0]:.3f})',
                   )
            
            # Set labels and formatting
            ax.set_xlabel(xlabel   = 'Day Index', 
                          fontsize = 12,
                         )

            ax.set_ylabel(ylabel   = 'Average PHQ-9 Score', 
                          fontsize = 12,
                         )

            ax.set_title(label    = 'Daily Average PHQ-9 Scores Over Time', 
                         fontsize = 16,
                        )

            ax.grid(True, alpha = 0.3)
            ax.legend()
            
            # Add severity level lines
            severity_levels = {'Minimal'  : 5, 
                               'Mild'     : 10, 
                               'Moderate' : 15,
                               'Severe'   : 20,
                              }

            colors          = ['green', 
                               'yellow', 
                               'orange', 
                               'red',
                              ]
            
            for (level, score), color in zip(severity_levels.items(), colors):
                ax.axhline(y         = score, 
                           color     = color, 
                           linestyle = ':', 
                           alpha     = 0.5, 
                           label     = f'{level} ({score})',
                          )
            
            ax.legend(bbox_to_anchor = (1.05, 1), 
                      loc            = 'upper left',
                     )

            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(fname       = save_path, 
                            dpi         = 300, 
                            bbox_inches = 'tight',
                           )
                self.logger.info(f"Daily average plot saved to: {save_path}")
            
            plt.show()
            
            # Log statistics
            self.logger.info(f"Daily average range: {daily_averages.min():.2f} - {daily_averages.max():.2f}")
            self.logger.info(f"Average trend slope: {z[0]:.3f} points per day")
            
        except Exception as e:
            self.logger.error(f"Error creating daily average plot: {e}")
            raise Exception(f"DailyAveragePlotError: {e}")

    
    def generate_comprehensive_report(self, output_dir: str = "analysis_results") -> dict:
        """
        Generate comprehensive analysis report with all visualizations and statistics
        
        Arguments:
        ----------
            output_dir { str } : Directory to save results
            
        Returns:
        --------
                { dict }       : Summary of analysis results
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded. Please load data first.")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents  = True, 
                              exist_ok = True,
                             )
            
            self.logger.info(f"Generating comprehensive report in: {output_path}")
            
            # Summary Statistics
            summary_stats = self.get_summary_statistics()
            summary_stats.to_csv(output_path / "summary_statistics.csv")
            
            # Scatter Plot
            self.create_scatter_plot(save_path = output_path / "scatter_plot.png")
            
            # Daily Average Plot
            self.create_daily_average_plot(save_path = output_path / "daily_averages.png")
            
            # Cluster Analysis
            elbow_point, optimal_clusters = self.find_optimal_clusters(save_plots = True, 
                                                                       plot_dir   = str(output_path),
                                                                      )
            
            # Fit clustering model with optimal clusters
            cluster_labels                = self.fit_clustering_model(optimal_clusters)
            
            # Plot clusters
            self.plot_clusters(cluster_labels, 
                               optimal_clusters, 
                               save_path = output_path / "cluster_results.png",
                              )
            
            # Cluster characteristics
            cluster_characteristics = self.analyze_cluster_characteristics(cluster_labels)
            cluster_characteristics.to_csv(output_path / "cluster_characteristics.csv", 
                                           index = False,
                                          )
            
            # Generate summary report
            report_summary = {'data_shape'                  : self.data.shape,
                              'total_scores'                : self.data.count().sum(),
                              'missing_percentage'          : (self.data.isna().sum().sum() / self.data.size) * 100,
                              'score_range'                 : f"{self.data.min().min():.1f} - {self.data.max().max():.1f}",
                              'elbow_point_clusters'        : elbow_point,
                              'optimal_clusters_silhouette' : optimal_clusters,
                              'cluster_summary'             : cluster_characteristics.to_dict('records'),
                             }
            
            # Save report summary
            with open(output_path / "analysis_summary.json", 'w') as f:
                json.dump(obj     = report_summary,
                          fp      = f, 
                          indent  = 4, 
                          default = str,
                         )
            
            self.logger.info("Comprehensive report generated successfully")
            
            return report_summary
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            raise Exception(f"ReportGenerationError: {e}")
    
    