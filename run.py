# DEPENDENCIES
import os
import json
import logging
import pandas as pd
from config import JUMP
from config import ALPHA
from config import PENALTY
from config import DATA_PATH
from config import MINIMUM_POINTS
from config import FINAL_PLOT_PATH
from config import RESULTS_FILE_PATH
from config import TEST_RESULTS_PATH
from config import VALIDATED_PLOT_PATH
from config import AGGREGATED_DATA_PATH
from config import AGGREGATED_PLOT_PATH
from config import SMOOTHING_WINDOW_SIZE
from config import RESULTS_BASE_DIRECTORY
from config import CLUSTER_BOUNDARIES_PATH
from src.change_point_detector import ChangePointDetector


# Set up centralized logging
os.makedirs("logs", exist_ok = True)

logging.basicConfig(level    = logging.INFO,
                    format   = "%(asctime)s - %(levelname)s - %(message)s",
                    handlers = [logging.FileHandler("logs/change_point_detection.log", mode = 'w'),
                                logging.StreamHandler(),
                               ],
                   )

# Results directory setup
os.makedirs(RESULTS_BASE_DIRECTORY, exist_ok = True)


logging.info("="*80)
logging.info("STARTING PHQ-9 CHANGE POINT DETECTION ANALYSIS")
logging.info("="*80)
        
# Initialize the Change Point Detector
logging.info("Initializing Change Point Detector...")
detector = ChangePointDetector(data_path             = DATA_PATH,
                               penalty               = PENALTY,
                               jump                  = JUMP,
                               minimum_points        = MINIMUM_POINTS,
                               smoothing_window_size = SMOOTHING_WINDOW_SIZE,
                               alpha                 = ALPHA,
                               final_plot_path       = FINAL_PLOT_PATH,
                               aggregated_plot_path  = AGGREGATED_PLOT_PATH,
                               validated_plot_path   = VALIDATED_PLOT_PATH,
                              )
        
logging.info(f"Configuration:")
logging.info(f"  - Data Path: {DATA_PATH}")
logging.info(f"  - Penalty Parameter: {PENALTY}")
logging.info(f"  - Jump Parameter: {JUMP}")
logging.info(f"  - Minimum Points: {MINIMUM_POINTS}")
logging.info(f"  - Smoothing Window Size: {SMOOTHING_WINDOW_SIZE}")
logging.info(f"  - Alpha (Significance Level): {ALPHA}")
        
# Run the complete analysis
logging.info("Running complete change point detection analysis...")
results  = detector.run_full_analysis()
        
# Log analysis results
logging.info("="*50)
logging.info("ANALYSIS RESULTS SUMMARY")
logging.info("="*50)
logging.info(f"Data shape: {results['raw_data_shape']}")
logging.info(f"Number of change points detected: {results['num_change_points']}")
logging.info(f"Change point indices: {results['change_points']}")
        
# Log cluster boundaries
logging.info("\nCluster Boundaries:")
for cluster_name, boundaries in results['cluster_boundaries'].items():
    logging.info(f"  {cluster_name}: Start = {boundaries['Start']}, End = {boundaries['End']}")
        
# Log statistical test results
logging.info("\nStatistical Test Results:")
for boundary_pair, test_info in results['statistical_test_results'].items():
    logging.info(f"  Boundaries {boundary_pair}:")
    logging.info(f"    Test: {test_info['test_name']}")
    logging.info(f"    P-value: {test_info['p-value']:.6f}")
    logging.info(f"    {test_info['interpretation']}")
        
# Log aggregated statistics
logging.info("\nAggregated Data Statistics:")
agg_stats = results['aggregated_statistics'].describe().to_dict()
logging.info(f"  Mean CV: {agg_stats['mean']:.4f}")
logging.info(f"  Std CV: {agg_stats['std']:.4f}")
logging.info(f"  Min CV: {agg_stats['min']:.4f}")
logging.info(f"  Max CV: {agg_stats['max']:.4f}")
        
    
# Convert results to JSON-serializable format
json_results = {'analysis_configuration'   : {'data_path'             : DATA_PATH,
                                              'penalty'               : PENALTY,
                                              'jump'                  : JUMP,
                                              'minimum_points'        : MINIMUM_POINTS,
                                              'smoothing_window_size' : SMOOTHING_WINDOW_SIZE,
                                              'alpha'                 : ALPHA,
                                             },
                'data_summary'             : {'raw_data_shape'    : results['raw_data_shape'],
                                              'num_change_points' : results['num_change_points'],
                                             },
                'change_points'            : results['change_points'],
                'cluster_boundaries'       : results['cluster_boundaries'],
                'aggregated_statistics'    : results['aggregated_statistics'],
                'statistical_test_results' : {},
               }
        
# Convert statistical test results to JSON-serializable format
for boundary_pair, test_info in results['statistical_test_results'].items():
    key                                           = f"{boundary_pair[0]}_to_{boundary_pair[1]}"

    json_results['statistical_test_results'][key] = {'boundary_1'           : boundary_pair[0],
                                                     'boundary_2'           : boundary_pair[1],
                                                     'test_name'            : test_info['test_name'],
                                                     'p_value'              : float(test_info['p-value']),
                                                     'test_statistic_value' : float(test_info['test_statistic_value']),
                                                     'interpretation'       : test_info['interpretation'],
                                                    }
        
with open(RESULTS_FILE_PATH, 'w') as f:
    json.dump(obj     = json_results, 
              fp      = f, 
              indent  = 4, 
              default = str,
             )
        
logging.info(f"\nDetailed results saved to: {RESULTS_FILE_PATH}")
        
# Save aggregated data to CSV
results['aggregated_statistics'].to_csv(path_or_buf = AGGREGATED_DATA_PATH, 
                                        header      = ['Coefficient_of_Variation'],
                                       )
                                       
logging.info(f"Aggregated CV data saved to: {AGGREGATED_DATA_PATH}")
        
# Save cluster boundaries to CSV
cluster_df = list()
for cluster_name, boundaries in results['cluster_boundaries'].items():
    cluster_df.append({'Cluster'   : cluster_name,
                       'Start_Day' : boundaries['Start'],
                       'End_Day'   : boundaries['End'],
                     })
        
       
pd.DataFrame(cluster_df).to_csv(CLUSTER_BOUNDARIES_PATH, index = False)
logging.info(f"Cluster boundaries saved to: {CLUSTER_BOUNDARIES_PATH}")
        
# Save statistical test results to CSV
test_df           = list()

for boundary_pair, test_info in results['statistical_test_results'].items():
    test_df.append({'Boundary_1_Start'  : boundary_pair[0][0],
                    'Boundary_1_End'    : boundary_pair[0][1],
                    'Boundary_2_Start'  : boundary_pair[1][0],
                    'Boundary_2_End'    : boundary_pair[1][1],
                    'Test_Name'         : test_info['test_name'],
                    'P_Value'           : test_info['p-value'],
                    'Test_Statistic'    : test_info['test_statistic_value'],
                    'Significant'       : test_info['p-value'] < ALPHA,
                    'Interpretation'    : test_info['interpretation'],
                  })
        
pd.DataFrame(test_df).to_csv(TEST_RESULTS_PATH, index = False)
logging.info(f"Statistical test results saved to: {TEST_RESULTS_PATH}")
        
logging.info("="*80)
logging.info("CHANGE POINT DETECTION ANALYSIS COMPLETED SUCCESSFULLY")
logging.info("="*80)
logging.info(f"Total files generated:")
logging.info(f"  - Plots: {FINAL_PLOT_PATH}, {AGGREGATED_PLOT_PATH}, {VALIDATED_PLOT_PATH}")
logging.info(f"  - Data files: {RESULTS_FILE_PATH}, {AGGREGATED_DATA_PATH}")
logging.info(f"  - Analysis files: {CLUSTER_BOUNDARIES_PATH}, {TEST_RESULTS_PATH}")
logging.info(f"Results saved in: {RESULTS_BASE_DIRECTORY}/")
logging.info("="*80)

