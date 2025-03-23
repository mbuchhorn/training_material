# -*- coding: utf-8 -*-
"""
SYNOPSIS
    fnc_Grdana.py parameters

DESCRIPTION
    This class run a inter-&intra outlier detection plus best feature selection. 
    
    The output are two txt-files:
        <input csv-file basename>-all-outliers.txt, this file contains the location_ids of the excluded data.
        <input csv-file basename>-bestbands.txt, this file contains the best bands sorted from best to worst. 

PREREQUISITES
    Python => 3.5
    
DEPENDENCIES
    cStatus_bar (if progress bar is requested)
    
AUTHOR
    Luc Bertels <luc.bertels@vito.be>
    Dr. Marcel Buchhorn <marcel.buchhorn@vito.be>

LICENSE
    This script is property of VITO. Copyrights or restrictions may apply.

VERSION
    4.2 (2022-08-23)
"""

import os
import time
import pandas as pd
import numpy as np
import warnings

class cGrdana():
    """MAIN CLASS to do the ground reference analysis"""
    def __init__(self, parameters):
        """init of class"""
        self.parameters = parameters

    def run(self):
        """run the analysis"""
        
        start_time = time.time()
        #only start the processing if the input check is successful
        if self.input_check():
            #get base file name from input CSV file and generate output file names
            sFile_basename = os.path.basename(self.ground_ref_file).split('.csv')[0]
            self.fBestFeatures = os.path.join(self.parameters['analysis_dir'], sFile_basename+'-bestbands.txt')
            self.fAllOutliers = os.path.join(self.parameters['analysis_dir'], sFile_basename+'-all-outliers.txt')

            print('** Processing file: ' + self.ground_ref_file)
            
            #run outlier detection if needed
            if self.parameters['outlier_detection']:
                print('** run the outlier detection...')
                print('*** retrieve Class/Metrics info for analysis...')
                self._1_Retrieve_Class_Info(self.ground_ref_file)
                print('*** calculate RMSE confusion matrix...')
                self._2_Calculate_Rule_MX()
                print('*** calculate class statistics before outlier removal...')
                self._3_Calculate_Stats('before')
                print(self.aClassInfo.to_string())

                if self.parameters['remove_intra_class_outliers']:
                    print('*** run the INTRA Class outlier detection...')
                    self._4_Exclude_Intra_Class_Outliers()
                
                print('*** run the INTER Class outlier detection...')
                self._5_Exclude_Inter_Class_Outliers()
                print('*** calculate RMSE confusion matrix...')
                self._2_Calculate_Rule_MX()
                print('*** calculate class statistics after outlier removal...')
                self._3_Calculate_Stats('after')
                print(self.aClassInfo.to_string())
                
                #create final outlier array
                if self.parameters['remove_intra_class_outliers']:
                    self.outlier_ids = self.ID_inter_outliers + self.ID_intra_outliers
                else:
                    self.outlier_ids = self.ID_inter_outliers
                
                #free resources
                self.dfValidData = None
                self.aClassInfo = None
                self.rule_MX = None
                
            # run Best Band selection per scenario if needed
            if self.parameters['select_best_bands']:
                print('** run the best bands selection...')
                print('*** retrieve Class/Metrics info for analysis...')
                self._1b_Retrieve_All_Class_Info(self.ground_ref_file)
                print('*** run the best band selection analysis...')
                self._6_Find_Best_Features()
            #write out results
            print('** write the results out')
            self._99_Output()
        else:
            raise
            
        print('** Analysing ground reference data finished. Needed time in Minutes: '),
        print("{:10.4f}".format((time.time() - start_time)/60))

    def input_check(self):
        """check the input file and output folder"""
        print('** run Input data check...')
        
        self.ground_ref_file = self.parameters['ground_ref_file']
        outdir = self.parameters['analysis_dir']
        
        #check if this file exists        
        if not os.path.exists(self.ground_ref_file):
            print('*** input file does not exist anymore.')
            return False
        #check if output directory is ready
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        
        return True

    def _1_Retrieve_Class_Info(self, ground_ref_file):
        """this function reads in the CSV file into a Pandas Dataframe
           and filter it to the given threshold"""
    
        # Read the ground reference data as a pandas dataframe
        df = pd.read_csv(ground_ref_file, dtype={'habitat': str})

        # Retrieve the features - i.e. the band names (bxx)
        self.features = self.parameters['selected_features']
        self.nFeatures = len(self.features) 

        # ini dataframe for final data
        dfValidData = pd.DataFrame()        
        # ini dataframe for class statistics
        aClassInfo = pd.DataFrame(columns=['class', 'nEntries', 'nExcluded', 'rmse_min_perc', 'rmse_median_perc'])
        
        #run over the input CSV file and read out the info for the selected classes for outlier detection       
        for forclass in self.parameters['selected_classes']:
            #in old LC100 algorithm we did the outlier detection only with the pure pixels (cover fraction threshold - to get a clean CL1 classification)
            #aNewDf = (df[df[forclass] >= float(self.parameters['minimum_cover_threshold'])]).sort_values(by=forclass, ascending=False)
            
            #for habitat we do not know the pureness of training point --> if too many outliers are detected play with THRESHOLD
            aNewDf = (df[df['habitat'] == forclass]).sort_values(by=['habitat'], ascending=False)
                        
            keys = ['entry'] + self.features 
            aNewDf = aNewDf[keys]
            
            if aNewDf.shape[0] > 0:
                #aNewDf.rename(columns = {forclass: 'coverage'}, inplace=True)
                aNewDf['class'] = forclass
                dfValidData = pd.concat([dfValidData, aNewDf], ignore_index=True)
                
                #aClassInfo = aClassInfo.append({'class': forclass, 'nEntries':aNewDf.shape[0], 'nExcluded':0, 'rmse_min_perc':0.0, 'rmse_median_perc':0.0}, ignore_index=True)
                new_row = pd.DataFrame([{'class': forclass, 'nEntries': aNewDf.shape[0], 'nExcluded': 0,
                                         'rmse_min_perc': 0.0, 'rmse_median_perc': 0.0}])
                aClassInfo = pd.concat([aClassInfo, new_row], ignore_index=True)
        #add the two cloumns for the outliers
        dfValidData = dfValidData.assign(inter_excluded=False, intra_excluded=False)
        #set index and sort the dataframes
        aClassInfo.set_index('class', inplace=True)        
        dfValidData.sort_values(by=['class'], inplace=True)
        dfValidData.set_index('entry', inplace=True)
       
        self.dfValidData = dfValidData
        self.aClassInfo = aClassInfo
    
    def _1b_Retrieve_All_Class_Info(self, ground_ref_file):
        """This functions read in the CSV file and uses all data, but
           prepare the dataframe in the same manner as in fuction _1_Retrieve_Class_Info"""
    
        # Read the ground reference data as a pandas dataframe:
        dfValidData = pd.read_csv(ground_ref_file,dtype={'habitat': str})

        # Retrieve the features - i.e. the band names (bxx)
        self.features = self.parameters['selected_features']
        self.nFeatures = len(self.features) 

        # set index and sort data
        dfValidData.sort_values(by=['habitat'], inplace=True)
        dfValidData.set_index('entry', inplace=True)
       
        self.dfAllValidData = dfValidData
    
    def _2_Calculate_Rule_MX(self):
        """this function calculates the RMSE confusion matrix for the
           input pandas dataframe"""

        #filter the dataframe for the data without outliers
        dfValidData=self.dfValidData[(self.dfValidData['intra_excluded'] == False) & \
                                     (self.dfValidData['inter_excluded'] == False)]
        #save the order of filtered dataframe (order of locations) for later reconstruction of this dataframe
        self.SpectraOrder = dfValidData.index.tolist()
        #get number of training points (spectras)
        nSpectra = dfValidData.shape[0]
        
        #get the feature data (metrics) for each training point (spectrum)
        OrigSpec = dfValidData[self.features].values.astype(np.float32)
        #set nodata values to NAN
        OrigSpec[OrigSpec == self.parameters['nodata_value']] = np.nan
        #scale the data between minimum and maximum values within the features (metric bands)
        #X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        minFeature = np.nanmin(OrigSpec , axis=0)
        maxFeature = np.nanmax(OrigSpec , axis=0)
        rangeFeature = maxFeature - minFeature
        #to avoid issues with zeros in devide
        rangeFeature[rangeFeature == 0.0] = 1.0
        
        NormData = (OrigSpec - minFeature ) / rangeFeature

        #ini the array to hold the rmse confusion matrix of all spectra (training points) against each other
        self.rule_MX = np.zeros((nSpectra, nSpectra), dtype=np.float32)
        #loop over all training points (spectra) to generate rmse between the spectras
        for iR in range(nSpectra):
            self.rule_MX[iR, iR] = np.nan
            
            for iC in range (iR+1, nSpectra):
                rmse = np.sqrt((np.power((NormData[iR] - NormData[iC]), 2)).sum() / self.nFeatures)
        
                self.rule_MX[iR, iC] = rmse
                self.rule_MX[iC, iR] = rmse

    def _3_Calculate_Stats(self, fStage):
        """This function calculates the class statistics out of the 
           RMSE confusion matrix"""
        
        #get the dataframe in the exact order as the self.rule_MX array!!!!
        #important - otherwise we get a mismatch
        dfValidData = self.dfValidData.loc[self.SpectraOrder].copy()
        dfValidData.reset_index(inplace=True)
   
        #loop over the classes and claculate the class statistics
        for forclass in self.parameters['selected_classes']:
            
            if not forclass in dfValidData['class'].unique().tolist():
                continue
            
            #get the row indices for the own class and all the other classes 
            # (these indices correspond to position in self.rile_MX)
            iOwn = dfValidData[dfValidData['class'] == forclass].index
            iOther = dfValidData[dfValidData['class'] != forclass].index
            
            #ini the arrays to hold the RMSE results
            nOwn = iOwn.shape[0]
            aRMSE_min = np.zeros((nOwn), dtype=float)
            aRMSE_median = np.zeros((nOwn), dtype=float)
            
            nPixels = nOwn**2 - nOwn

            # run over all spectra (trainign points) in the OWN class
            for iX in iOwn:
                # Get indices to valid in-class spectra (training points)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    iRMSE_min     = np.less(self.rule_MX[iOwn, iX], np.nanmin(self.rule_MX[iOther, iX]))
                    iRMSE_median  = np.less(self.rule_MX[iOwn, iX], np.nanmedian(self.rule_MX[iOther, iX]))
            
                #Count the valid in-class spectra:
                aRMSE_min[iRMSE_min] += 1
                aRMSE_median[iRMSE_median] +=1                
            
            #now calculate the statistics for the whole class
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.aClassInfo.at[forclass, 'rmse_min_perc'] = aRMSE_min.sum() / nPixels * 100
                self.aClassInfo.at[forclass, 'rmse_median_perc'] = aRMSE_median.sum() / nPixels * 100
        
        #write out
        self.aClassInfo.to_csv(os.path.join(self.parameters['analysis_dir'], 'Class_statistics_{}_outlier-removal.csv'.format(fStage)))


    def _4_Exclude_Intra_Class_Outliers(self):
        """this function calculate the INTRA class outliers
           and report the location_id of these outliers"""

        #get the dataframe in the exact order as the self.rule_MX array!!!!
        dfValidData = self.dfValidData.loc[self.SpectraOrder].copy()
        dfValidData.reset_index(inplace=True)
        #ini the lists for the location_ids of the outlier and the deltas for these outliers
        nIntra_excluded = 0
        aID = []
        aDelta = []
        
        #loop over each class
        for forclass in self.parameters['selected_classes']:
            
            # check if the class exist:
            if not forclass in dfValidData['class'].unique().tolist():
                continue
            
            #get the row indices for all training points in the current class
            # (these indices correspond to position in self.rile_MX)
            iOwn = dfValidData[dfValidData['class'] == forclass].index
 
            # Calculate classmedian
            aOwn_MX = np.zeros((iOwn.shape[0],iOwn.shape[0]), dtype=np.float32)
            for iX in range(iOwn.shape[0]):
                aOwn_MX[iX,:] = self.rule_MX[iOwn, iOwn[iX]]
            Class_Median_RMSE = np.nanmedian(aOwn_MX)
            
            #calculate median delta from threshold for outlier detection
            Class_Median_delta = Class_Median_RMSE * (self.parameters['intra_rmse_threshold'] - 1.0)

            # Handle all in class indices:
            for iX in range(iOwn.shape[0]):
                #check the training point RMSE against class median threshold --> INTRA outlier check
                if (np.nanmedian(aOwn_MX[:, iX]) < (Class_Median_RMSE - Class_Median_delta)) | \
                (np.nanmedian(aOwn_MX[:, iX]) > (Class_Median_RMSE + Class_Median_delta)):  

                    #get the location_ID of this training point (outlier) and the delta in RMSE
                    id=dfValidData.location_id.iloc[iOwn[iX]]
                    aID.append(id)
                    # the delta has to be correctly calculated between class median and point median to others
                    aDelta.append(np.ptp([np.nanmedian(aOwn_MX[:, iX]), Class_Median_RMSE]))
                    
        ##now we weight the importance of the found outliers against the "to report outlier threshold"
        #calculate weight factor for each outlier by taken the DELTA into account
        aDeltaMax = max(aDelta)
        aWeight = [x/float(aDeltaMax) for x in aDelta]
        #get the outlier we want to report
        aKeep = [i for i, j in enumerate(aWeight) if j > self.parameters['outlier_weight_threshold']]
        aValidLocations = [aID[i] for i in aKeep]
        self.ID_intra_outliers = aValidLocations
                    
        #run loop to get final selected INTRA outliers and perform changes on master dataframes
        for element in aValidLocations:
            self.dfValidData.at[element, 'intra_excluded'] = True
            nIntra_excluded += 1
            forClass = self.dfValidData.loc[element, 'class']
            self.aClassInfo.at[forClass, 'nExcluded'] = self.aClassInfo.loc[forClass, 'nExcluded'] + 1

        print('**** Overall ' + str(nIntra_excluded) + ' intra-specific outliers were removed ...')      
        if len(self.ID_intra_outliers) > 0:
            print('**** location_ids: ' + ', '.join([str(x) for x in self.ID_intra_outliers]))
            
    def _5_Exclude_Inter_Class_Outliers(self):
        """this function calculate the INTER class outliers
           and report the location_id of these outliers"""

        #get the dataframe in the exact order as the self.rule_MX array!!!!
        dfValidData = self.dfValidData.loc[self.SpectraOrder].copy()
        dfValidData.reset_index(inplace=True)
        #ini the lists for the location_ids of the outlier and the deltas for these outliers
        nInter_excluded = 0
        aID = []
        aDelta = []
        
        #loop over each class
        for forclass in self.parameters['selected_classes']:
            
            # check if the class exist:
            if not forclass in dfValidData['class'].unique().tolist():
                continue
            
            #get the row indices for all training points in the current class and other classes
            # (these indices correspond to position in self.rile_MX)
            iOwn = dfValidData[(dfValidData['class'] == forclass) & (dfValidData['intra_excluded'] == False)].index
            iOther = dfValidData[(dfValidData['class'] != forclass) & (dfValidData['intra_excluded'] == False)].index

            # Handle all in class indices:
            for iX in iOwn:
                #check the training point RMSE against class median threshold of other classes--> INTER outlier check
                if np.nanmedian(self.rule_MX[iX, iOwn]) > (np.nanmedian(self.rule_MX[iX, iOther]) * self.parameters['inter_rmse_threshold']):
                    #get the location_ID of this training point (outlier) and the delta in RMSE
                    id=dfValidData.location_id.iloc[iX]
                    aID.append(id)
                    
                    # delta calculation
                    aDelta.append(np.nanmedian(self.rule_MX[iX, iOwn]) - (np.nanmedian(self.rule_MX[iX, iOther])))

        ##now we weight the importance of the found outliers against the "to report outlier threshold"
        #calculate weight factor for each outlier by taken the DELTA into account
        aDeltaMax = max(aDelta)
        aWeight = [x/float(aDeltaMax) for x in aDelta]
        #get the outlier we want to report
        aKeep = [i for i, j in enumerate(aWeight) if j > self.parameters['outlier_weight_threshold']]
        aValidLocations = [aID[i] for i in aKeep]
        self.ID_inter_outliers = aValidLocations
        
        #run loop to get final selected INTER outliers and perform changes on master dataframes
        for element in aValidLocations:
            self.dfValidData.at[element, 'inter_excluded'] = True
            nInter_excluded += 1
            forClass = self.dfValidData.loc[element, 'class']
            self.aClassInfo.at[forClass, 'nExcluded'] = self.aClassInfo.loc[forClass, 'nExcluded'] + 1

        print('**** Overall ' + str(nInter_excluded) + ' inter-specific outliers were removed ...')   
        if len(self.ID_inter_outliers) > 0:
            print('**** location_ids: ' + ', '.join([str(x) for x in self.ID_inter_outliers]))
            
    def _6_Find_Best_Features(self):
        """this function runs the best band selection for each given scenario
           on all the input data with outliers removed if checked"""

        #ini dictionary to hold the final best bands for each scenario     
        self.BestFeatures = {}
        
        #if outlier detection was run on the pure samples then remove these now from the dataframe of all data
        print('**** entries in Master Table: ' + str(self.dfAllValidData.shape[0]))
        if self.parameters['outlier_detection']:
            self.dfAllValidData = self.dfAllValidData[~self.dfAllValidData.index.isin(self.outlier_ids)]
            print('**** entries in Master Table after outlier removal: ' + str(self.dfAllValidData.shape[0]))

        #generate the correct dataframe for this scenario (plus generate the needed class column)
        dfValidData = self.dfAllValidData.copy()
        dfValidData['class'] = dfValidData['habitat']
        dfValidData.reset_index(inplace=True)
        
        #get the class names for this scenario for the best band comparison
        aClasses = dfValidData['class'].unique().tolist()
        nClasses = len(aClasses)
        
        #ini the arrays for the statistics of each feature (metrics band)
        nCombinations = ((nClasses * nClasses) - nClasses) / 2
        aStatsAna       = np.zeros((self.nFeatures, int(nCombinations)), dtype=np.float32)
        aStatsAna[:,:]  = np.nan
        aStatsResult    = np.zeros((self.nFeatures), dtype=np.float32)      
        iFeature = -1   #makes sure we have the metrics band order correct in the statistics
        
        #For each feature (metric band) calculate the class overlap
        for feature in self.features:
            print('**** analysis for feature: {}'.format(feature))
            iFeature += 1
            iD = 0

            # Handle all class combinations
            for iCs in range(nClasses-1):
                #get the class name which acts as reference in the comparison (e.g. grass)
                class_ref = aClasses[iCs]
                #get the reference class data for this feature (metric band)
                aClass_Ref = dfValidData[dfValidData['class'] == class_ref][feature].values
                if aClass_Ref.shape[0] == 0:
                    continue
                 
                for iCe in range(iCs+1, nClasses):
                    #get the class name which acts as target in the comparison (e.g. tree)
                    class_tar = aClasses[iCe]
                    #get the target class data for this feature (metric band)
                    aClass_Tar = dfValidData[dfValidData['class'] == class_tar][feature].values
                    if aClass_Tar.shape[0] == 0:
                        continue
                                       
                    #run the analysis for this reference-target-class combo and this specific feature (metric band)
                    #if we didn't run the outlierselection it is better to run the feature selection with percentile
                    if self.parameters['best_band_by_percentile']:
                        aMin  = [np.nanpercentile(aClass_Ref, 2), np.nanpercentile(aClass_Tar, 2)]
                        aMax  = [np.nanpercentile(aClass_Ref, 98), np.nanpercentile(aClass_Tar, 98)]
                    else:
                        aMin  = [min(aClass_Ref), min(aClass_Tar)]
                        aMax  = [max(aClass_Ref), max(aClass_Tar)]
                    
                    #run seperability
                    Min_v = max(aMin)
                    Max_v = min(aMax)
                    nRef  = np.sum((aClass_Ref < Min_v) | (aClass_Ref > Max_v))
                    nTar  = np.sum((aClass_Tar < Min_v) | (aClass_Tar > Max_v))
                    #save the result of this class-combo separability for this feature (metric band)
                    aStatsAna[iFeature, iD] = float(nRef + nTar) / (aClass_Ref.shape[0] + aClass_Tar.shape[0])
                    
                    #print('***** separability for class {} versus {}: {}'.format(class_ref, class_tar, aStatsAna[iFeature, iD]))
                    iD += 1
            # save the statistics for the whole feature (metric band) by calculating median of all class-combos (separability indicator)  
            print('***** separability score over all classes: {}'.format(np.mean(aStatsAna[iFeature, :])))
            aStatsResult[iFeature] = np.mean(aStatsAna[iFeature, :])
        
        # analyze the separability indicators by sorting them highest to lowest 
        iBestFeatures   = aStatsResult.argsort()[::-1]
        # get the best band names in the right order
        BestFeatures    = []
        for i in iBestFeatures:
            BestFeatures.append(self.features[i])
        #get the score
        Score = []
        for i in iBestFeatures:
            Score.append(aStatsResult[i])
         
        #write to dictionary
        self.BestFeatures['bands'] = BestFeatures
        self.BestFeatures['Score'] = Score
        print('**** BestBands: ' + ','.join(BestFeatures))
                                
    def _99_Output(self):
        """this function saves the results to disk"""
        
        # save the detected outliers to disk
        if self.parameters['outlier_detection']:
            with open(self.fAllOutliers, 'w') as f:
                if self.parameters['outlier_detection']:
                    f.write(','.join([str(x) for x in self.outlier_ids]) + '\n')

        #save the detected BestBands to disk
        if self.parameters['select_best_bands']:
            with open(self.fBestFeatures, 'w') as f:
                for BestFeature in self.BestFeatures: 
                    f.write(BestFeature + ': ' + ','.join([str(x) for x in self.BestFeatures[BestFeature]]) + '\n')
