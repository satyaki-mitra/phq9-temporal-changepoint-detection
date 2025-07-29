# DEPENDENCIES
import os
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime


# DATA GENERATION CLASS
class PHQ9DataGenerator:
    def __init__(self, total_patients: int = 100, total_days: int = 365, required_sample_count: int = 50, maximum_surveys_attempted: int = 7, seed: int = 2023):
        """
        Initialize the PHQ-9 data generator
        """
        self.total_patients            = total_patients
        self.total_days                = total_days
        self.required_sample_count     = required_sample_count
        self.maximum_surveys_attempted = maximum_surveys_attempted
        self.seed                      = seed
        self.logger                    = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set random seeds
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.logger.info(f"Initialized PHQ9DataGenerator | patients = {self.total_patients}, days = {self.total_days}, max_surveys = {self.maximum_surveys_attempted}, seed = {self.seed}")


    def _generate_phq9_samples(self, time_index: int):
        """
        Generate PHQ-9 score based on the time index using probability distributions that mimic depression trends
        
        Arguments:
        ----------
            time_index { int } : Day number (1-365) representing treatment progression
            
        Returns:
        --------
            { np.ndarray }     : Array containing a single PHQ-9 score
        """
        try:
            # Define score ranges and probabilities based on time progression
            if (1 <= time_index <= 5):
                # Early treatment: High scores (severe depression)
                upper_bound   = 27
                lower_bound   = 10
                probabilities = [0.056, 0.056, 0.056, 0.056, 0.028, 0.028, 0.028, 0.028, 0.014, 0.014, 0.014, 0.014, 0.007, 0.007, 0.007, 0.0035, 0.0035, 0.00175]
                
            elif (6 <= time_index <= 10):
                # Early-mid treatment: Moderate-severe scores
                upper_bound   = 19
                lower_bound   = 10
                probabilities = [0.1] * 10
                
            elif (11 <= time_index <= 20):
                # Mid treatment: Moderate scores with some improvement
                upper_bound   = 19
                lower_bound   = 5
                probabilities = [0.06, 0.06, 0.06, 0.03, 0.03, 0.03, 0.015, 0.015, 0.015, 0.0075, 0.0075, 0.0075, 0.00375, 0.00375, 0.00375]
                
            elif (21 <= time_index <= 30):
                # Mid-late treatment: Continued improvement
                upper_bound   = 19
                lower_bound   = 0
                probabilities = [0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.00160625, 0.000803125, 0.0004015625, 0.00020078125, 0.000100390625] + [5.01953125e-5] * 10
                
            elif (31 <= time_index <= 40):
                # Late treatment: Lower scores
                upper_bound   = 14
                lower_bound   = 0
                probabilities = [0.06667, 0.03333, 0.01667, 0.00833, 0.004167, 0.00208, 0.00104, 0.00052, 0.00026, 0.00013, 6.51041e-5, 3.25521e-5, 1.62760e-5, 8.13802e-6, 4.06901e-6]
                
            elif (41 <= time_index <= 100):
                # Extended treatment: Continued low scores
                upper_bound   = 14
                lower_bound   = 0
                probabilities = [0.06667, 0.03333, 0.01667, 0.00833, 0.004167, 0.00208, 0.00104, 0.00052, 0.00026, 0.00013, 6.51041e-5, 3.25521e-5, 1.62760e-5, 8.13802e-6, 4.06901e-6]
                
            elif (101 <= time_index <= 220):
                # Long-term follow-up: Mild scores
                upper_bound   = 9
                lower_bound   = 0
                probabilities = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025]
                
            elif (221 <= time_index <= 300):
                # Extended follow-up: Continued mild scores
                upper_bound   = 9
                lower_bound   = 0
                probabilities = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025]
                
            elif (301 <= time_index <= 365):
                # Long-term maintenance: Minimal scores
                upper_bound   = 4
                lower_bound   = 0
                probabilities = [0.5, 0.25, 0.15, 0.1, 0.05]

            else:
                self.logger.warning(f"Time index {time_index} is out of expected range")
                return np.array([np.nan])
            
            # Generating a list of all possible cases of PHQ-9 scores i.e Population within the possible range of variation 
            population       = list(np.arange(start = lower_bound, 
                                              stop  = (upper_bound+1), 
                                              step  = 1, 
                                              dtype = int,
                                             ),
                                   )
            
            # Normalize probabilities
            normalized_probs = [prob/sum(probabilities) for prob in probabilities]
            
            # Now draw sample of specified size from the above generated population and drawing should be done using Simple
            # Random Sampling with replacement as more than one person may have same PHQ-9 score
            required_sample  = np.random.choice(a       = population, 
                                                replace = True, 
                                                size    = 1, 
                                                p       = normalized_probs,
                                               )
            
            return required_sample

        except Exception as e:
            self.logger.error(f"Error generating sample for index {time_index}: {e}")
            return None
    

    def sample_allocations(self) -> tuple:
        """
        Sample allocation of patients to days with a probability model
            
        Returns:
        --------
                     { tuple }            : A python tuple containing patients list and desired sampled days with probabilistic allocations
        """
        if (self.required_sample_count > self.total_patients):
            self.logger.error("Requested sample count exceeds total patients")
            return repr(ValueError('Total number of patients should be greater than required sample count'))
        
        try:
            # Make a list of all possible timestamps
            days_list           = list(np.arange(start = 1,
                                                 stop  = (self.total_days+1),
                                                 step  = 1,
                                                 dtype = int,
                                                ),
                                      )
            
            patients_list       = list(np.arange(start = 1,
                                                 stop  = (self.total_patients+1),
                                                 step  = 1,
                                                 dtype = int,
                                                ),
                                      )
            
            # Assign probabilities for each day of getting selected
            probabilities       = ([0.99] * 1 + 
                                   [0.85] * 4 + 
                                   [0.85] * 10 + 
                                   [0.7] * 5 + 
                                   [0.5] * 20 + 
                                   [0.3] * 20 + 
                                   [0.25] * 60 + 
                                   [0.125] * 60 + 
                                   [0.0625] * 60 + 
                                   [0.03125] * 60 + 
                                   [0.015625] * 62 + 
                                   [0.95] * 3
                                  )
            
            # Select desired number of samples randomly from the all possible cases
            desired_sample_days = sorted(np.random.choice(a       = days_list,
                                                          size    = self.required_sample_count,
                                                          replace = False,
                                                          p       = [probability / sum(probabilities) for probability in probabilities],
                                                         ),
                                        )

            self.logger.info(f"Sampled {self.required_sample_count} days for survey allocations")

            return patients_list, desired_sample_days 
        
        except Exception as SampleAllocationError:
            self.logger.error(f"Error during sample allocation: {repr(SampleAllocationError)}")
            raise
    

    def create_dataset(self, desired_sample_days: list) -> pd.DataFrame:
        """
        Create PHQ-9 synthetic dataset for selected days
        """
        try:
            self.logger.info("Starting dataset generation...")

            # Create an empty dataframe
            output_dataframe = pd.DataFrame(index = [f"Day_{day}" for day in desired_sample_days])
            
             # Generate data for each patient
            for day in desired_sample_days:
                # Calculate the number of patients attempting surveys for the given day
                patients_attempting_survey = int(self.total_patients * (2 ** (-day / 365)))

                # Randomly select the patients attempting surveys
                random_patient_attempting  = sorted(random.sample(range(1, self.total_patients + 1), patients_attempting_survey))

                # Generate PHQ-9 scores for each patient
                for patient in random_patient_attempting:
                    # Randomly select the number of surveys attempted by the patient
                    num_surveys = random.randint(1, self.maximum_surveys_attempted)
                    
                    for survey_count in range(num_surveys):
                        # Skip adding score if survey_count exceeds "maximum_surveys_attempted"
                        if (survey_count >= self.maximum_surveys_attempted):
                            continue
                        
                        # Calculate the PHQ-9 score
                        score       = self._generate_phq9_samples(time_index = day)
                        column_name = f"Patient_{patient}"
                        
                        # Add the score to the dataframe
                        output_dataframe.loc[f"Day_{day}", column_name] = score[0]
            
            # For each patients' select only surveys attempted between a defined range
            for column in output_dataframe.columns:

                # Get the non-null values in the column
                non_nan_values = output_dataframe[column].dropna().values

                 # Check if there are any non-null values
                if (len(non_nan_values) > 0):
                    # Randomly select a range between 1 and 7
                    num_cells                                                                = np.random.randint(self.maximum_surveys_attempted - 2, self.maximum_surveys_attempted)
                    
                    # Randomly select indices from non-null values
                    selected_indices                                                         = np.random.choice(a       = len(non_nan_values), 
                                                                                                                size    = num_cells, 
                                                                                                                replace = False,
                                                                                                               )

                    # Create a mask with True values for the selected indices                                                                                           
                    mask                                                                     = np.zeros(len(output_dataframe), dtype=bool)
                    mask[np.flatnonzero(output_dataframe[column].notna())[selected_indices]] = True

                    # Set all other cells in the column as NaN
                    output_dataframe.loc[~mask, column]                                      = np.nan

            # Set the index name
            output_dataframe.index.name = 'Day'
            column_order                = [f'Patient_{i}' for i in range(1, self.total_patients + 1)]
            final_output_dataframe      = output_dataframe[column_order]

            self.logger.info("Dataset creation completed successfully !")
            return final_output_dataframe

        except Exception as e:
            self.logger.error(f"Error in create_dataset: {e}")
            raise


    def save_to_csv(self, dataframe: pd.DataFrame, filename: str):
        """
        Save the generated dataset to a CSV file.
        """
        try:
            dataframe.to_csv(path_or_buf = filename, 
                             index       = True,
                            )

            self.logger.info(f"Dataset successfully saved to {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save dataset to CSV: {e}")
            raise


