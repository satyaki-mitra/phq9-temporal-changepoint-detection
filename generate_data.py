# DEPENDENCIES
import os
import logging
from src.synthetic_phq9_data_generator import PHQ9DataGenerator

# Set up centralized logging here 
os.makedirs("logs", exist_ok = True)

logging.basicConfig(level    = logging.INFO,
                    format   = "%(asctime)s - %(levelname)s - %(message)s",
                    handlers = [logging.FileHandler("logs/phq9_data_generator.log", mode = 'w'), 
                                logging.StreamHandler(),
                               ],
                   )

# Configuration Variables
total_patients            = 100
total_days                = 365
required_sample_count     = 50
maximum_surveys_attempted = 7
random_seed               = 2023

datafile_destination      = "data/synthetic_phq9_data.csv"


# Initialize the data generation class 
data_generator            = PHQ9DataGenerator(total_patients            = total_patients,
                                              total_days                = total_days,
                                              required_sample_count     = required_sample_count,
                                              maximum_surveys_attempted = maximum_surveys_attempted,
                                              seed                      = random_seed,
                                             )

patients, sample_days     = data_generator.sample_allocations()

generated_dataframe       = data_generator.create_dataset(desired_sample_days = sample_days)

data_generator.save_to_csv(dataframe = generated_dataframe,
                           filename  = datafile_destination,
                          )
