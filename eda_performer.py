# DEPENDENCIES
import os
import logging
from src.phq9_data_analyzer import PHQ9DataAnalyzer


# Set up centralized logging here 
os.makedirs("logs", exist_ok = True)

logging.basicConfig(level    = logging.INFO,
                    format   = "%(asctime)s - %(levelname)s - %(message)s",
                    handlers = [logging.FileHandler("logs/phq9_exploratory_data_analysis.log", mode = 'w'), 
                                logging.StreamHandler(),
                               ],
                   )

# Configuration Variables
data_file_path          = "data/synthetic_phq9_data.csv"
results_base_directory  = "results/eda_results"
figure_size             = (15, 12)
max_clusters_to_test    = 20
clustering_random_seed  = 1234

# Results directory setup
os.makedirs(results_base_directory, 
            exist_ok = True)

# Initialize the PHQ-9 data analyzer
analyzer = PHQ9DataAnalyzer(data_path = data_file_path,
                            figsize   = figure_size,
                           )

# Validate the loaded data
data_validation_passed = analyzer.validate_data()

if not data_validation_passed:
    logging.error("Data validation failed. Stopping analysis.")
    exit(1)

logging.info(f"Data loaded successfully with shape: {analyzer.data.shape}")

# Generate summary statistics
summary_statistics = analyzer.get_summary_statistics()

# Create scatter plot visualization
scatter_plot_path = f"{results_base_directory}/scatter_plot.png"
analyzer.create_scatter_plot(save_path = scatter_plot_path)

# Find optimal number of clusters
elbow_clusters, silhouette_clusters = analyzer.find_optimal_clusters(max_clusters = max_clusters_to_test,
                                                                     save_plots   = True,
                                                                     plot_dir     = results_base_directory,
                                                                    )

# Perform clustering analysis with optimal clusters
optimal_cluster_count                = silhouette_clusters
cluster_labels                       = analyzer.fit_clustering_model(n_clusters = optimal_cluster_count)

# Generate cluster visualization
cluster_plot_path                    = f"{results_base_directory}/cluster_results.png"

analyzer.plot_clusters(cluster_labels = cluster_labels,
                       n_clusters     = optimal_cluster_count,
                       save_path      = cluster_plot_path,
                      )

# Analyze cluster characteristics
cluster_analysis                     = analyzer.analyze_cluster_characteristics(cluster_labels = cluster_labels)

# Compare different clustering methods
clustering_comparison                = analyzer.compare_clustering_methods(max_clusters = 10)

# Generate comprehensive analysis report
comprehensive_report                 = analyzer.generate_comprehensive_report(output_dir = results_base_directory)

# Save additional analysis results
summary_statistics.to_csv(f"{results_base_directory}/summary_statistics.csv")
cluster_analysis.to_csv(f"{results_base_directory}/cluster_characteristics.csv", index = False)
clustering_comparison.to_csv(f"{results_base_directory}/clustering_method_comparison.csv", index = False)

logging.info("="*80)
logging.info("PHQ-9 ANALYSIS COMPLETED SUCCESSFULLY")
logging.info("="*80)
logging.info(f"Data shape: {analyzer.data.shape}")
logging.info(f"Total PHQ-9 scores: {analyzer.data.count().sum()}")
logging.info(f"Missing data percentage: {(analyzer.data.isna().sum().sum() / analyzer.data.size) * 100:.2f}%")
logging.info(f"Score range: {analyzer.data.min().min():.1f} - {analyzer.data.max().max():.1f}")
logging.info(f"Elbow method suggests: {elbow_clusters} clusters")
logging.info(f"Silhouette analysis suggests: {silhouette_clusters} clusters")
logging.info(f"Using {optimal_cluster_count} clusters for final analysis")
logging.info(f"Results saved in: {results_base_directory}/")
logging.info("="*80)