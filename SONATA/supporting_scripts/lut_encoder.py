import pandas as pd
import numpy as np

# Script ensures that the models can work with categorical habitat labels by encoding them into numerical values during training and decoding them back for evaluation and interpretation.
class HabitatEncoder:
    def __init__(self, eunis_lut_path): # initializes the encoder with the path to a lookup table containing mappings between EUNIS codes and raster values
        self.eunis_lut = pd.read_excel(eunis_lut_path)
        self.label_to_int = {}
        self.int_to_label = {}
        self.classes_ = []

    def fit(self, labels): # takes a list of habitat labels and extracts relevant information from the EUNIS lookup table to create mappings between habitat labels and numerical values.
        mapping_df = self.eunis_lut[self.eunis_lut['eunis_code'].isin(
            labels)].loc[:, ['raster_value', 'eunis_code']].reset_index(drop=True)
        for row in mapping_df.itertuples():
            self.label_to_int[row.eunis_code] = row.raster_value
            self.int_to_label[row.raster_value] = row.eunis_code
        self.classes_ = np.array(mapping_df['eunis_code'].values)
        self.eunis_lut = None

    def transform(self, labels): # Convert list of habitat labels to corresponding numerical values using the mappings created in 'fit'
        if len(labels.shape) > 1:
            labels = np.squeeze(labels)
        labels = list(labels)
        encoded_labels = []
        for label in labels:
            encoded_labels.append(self.label_to_int[label])
        return np.array(encoded_labels)

    def inverse_transform(self, encoded_labels): # Convert a list of numerical values back to their original habitat labels using the reverse mappings
        if len(encoded_labels.shape) > 1:
            encoded_labels = np.squeeze(encoded_labels)
        encoded_labels = list(encoded_labels)
        decoded_labels = []
        for encoded_label in encoded_labels:
            decoded_labels.append(self.int_to_label[encoded_label])
        return np.array(decoded_labels)
