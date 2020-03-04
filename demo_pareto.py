# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:41:53 2020

@author: tlac980
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rn
from kneed import KneeLocator




class DemoPareto():
    
    def __init__(self,
                 score = np.array([[0.857691,0.0173355],
                   [0.844025,0.016638],
                   [0.8643315,0.02151],
                   [0.85958,0.026497],
                   [0.866869,0.028566],
                   [0.852132,0.030129],
                   [0.8638795,0.0373155],
                   [0.864325,0.0404965],
                   [0.8548655,0.0424155],
                   [0.8651765,0.049121],
                   [0.863405,0.05028],
                   [0.8701275,0.0589905],
                   [0.8543425,0.0579445],
                   [0.8671825,0.071308]]),
                 names = ['ARF65','ARF70','ARD75','ARF80','ARF85','ARF90','ARF95','ARF100','ARF105','ARF110','ARF115','ARF120','ARF125','ARF130'],
                 perc_redund = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
        
        self.score = score
        self.names = names
        self.perc_redund = perc_redund
        
        
    def plot_data(self) :
        x_all = self.score[:, 0]
        y_all = self.score[:, 1]
        
        plt.plot(x_all, y_all, 'ko')

        for i, txt in enumerate(self.names):
            plt.annotate(txt, (x_all[i], y_all[i]))
        
        plt.title('Pareto front')
        plt.xlabel('Kappa')
        plt.ylabel('RAM hours')
        xmin, xmax, ymin, ymax = plt.axis()
        plt.show()

        
    def draw_pareto(self) :
       
        # Calculate pareto front
        pareto = self.identify_pareto(self.score)
        #print ('Pareto front index vales')
        #print ('Points on Pareto front: \n',pareto)
        
        pareto_front = self.score[pareto]
        #print ('\nPareto front scores')
        #print (pareto_front)
        
        pareto_front_df = pd.DataFrame(pareto_front)
        pareto_front_df.sort_values(0, inplace=True)
        pareto_front = pareto_front_df.values
        
        x_all = self.score[:, 0]
        y_all = self.score[:, 1]
        
        x_pareto = pareto_front[:, 0]
        y_pareto = pareto_front[:, 1]
        
        plt.plot(x_all, y_all, 'ko')

        for i, txt in enumerate(self.names):
            plt.annotate(txt, (x_all[i], y_all[i]))
        
        plt.title('Pareto front')
        plt.plot(x_pareto, y_pareto, color='r')
        plt.xlabel('Kappa')
        plt.ylabel('RAM hours')
        xmin, xmax, ymin, ymax = plt.axis()
        plt.show()
        
    def draw_knee(self) :
        
        # Calculate pareto front
        pareto = self.identify_pareto(self.score)
        #print ('Pareto front index vales')
        #print ('Points on Pareto front: \n',pareto)
        
        pareto_front = self.score[pareto]
        #print ('\nPareto front scores')
        #print (pareto_front)
        
        pareto_front_df = pd.DataFrame(pareto_front)
        pareto_front_df.sort_values(0, inplace=True)
        pareto_front = pareto_front_df.values
        
        x_all = self.score[:, 0]
        y_all = self.score[:, 1]
        x_pareto = pareto_front[:, 0]
        y_pareto = pareto_front[:, 1]
        
        scorepd = pd.DataFrame(self.score,columns = ['X' , 'Y'])
        
        # Detect Knee point on the pareto
        try :
            kn = KneeLocator(x_pareto, y_pareto, curve='convex', direction='increasing',S=0)
            # Knee variable is used 
            kneeX = kn.knee
            kneeY = y_pareto[np.where(x_pareto == kneeX)[0][0]]
            
            # Get the index of the selected configuration 
            idName = scorepd.loc[(scorepd['X'] == kneeX) & (scorepd['Y'] == kneeY)].index[0]
            
            
        except (IndexError, ValueError)  :
            try :
                kn = KneeLocator(x_pareto, y_pareto, curve='concave', direction='increasing',S=0)
                kneeX = kn.knee
                kneeY = y_pareto[np.where(x_pareto == kneeX)[0][0]]
                
                # Get the index of the selected configuration 
                idName = scorepd.loc[(scorepd['X'] == kneeX) & (scorepd['Y'] == kneeY)].index[0]
                
            except (IndexError, ValueError) :
                kneeX = pareto_front[len(pareto_front)-1][0]
                if all(x == x_pareto[0] for x in x_pareto) :
                    kneeY = pareto_front[np.argmin(pareto_front.T[1][:])][1]
                else :
                    kneeY = scorepd.loc[(scorepd['X'] == kneeX)].iloc[0]['Y']
                # Get the index of the selected configuration 
                idName = scorepd.loc[(scorepd['X'] == kneeX) & (scorepd['Y'] == kneeY)].index[0]
        
        plt.plot(x_all, y_all, 'ko')
        
        for i, txt in enumerate(self.names):
            plt.annotate(txt, (x_all[i], y_all[i]))
        
        plt.title('Pareto front')
        plt.plot(x_pareto, y_pareto, color='r')
        plt.xlabel('Kappa')
        plt.ylabel('RAM hours')
        xmin, xmax, ymin, ymax = plt.axis()
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
        plt.show()
        
    
    def identify_pareto(self, scores):
        # Count number of items
        population_size = scores.shape[0]
        # Create a NumPy index for scores on the pareto front (zero indexed)
        population_ids = np.arange(population_size)
        # Create a starting list of items on the Pareto front
        # All items start off as being labelled as on the Parteo front
        pareto_front = np.ones(population_size, dtype=bool)
        # Loop through each item. This will then be compared with all other items
        for i in range(population_size):
            # Loop through all other items
            for j in range(population_size):
                # Check if our 'i' pint is dominated by out 'j' point
                if (scores[j][0] >= scores[i][0]) and (scores[j][1] <= scores[i][1]) and (scores[j][0] > scores[i][0]) and (scores[j][1] < scores[i][1]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
        # Return ids of scenarios on pareto front
        return population_ids[pareto_front]

    def calculate_crowding(self, scores):
        # Crowding is based on a vector for each individual
        # All dimension is normalised between low and high. For any one dimension, all
        # solutions are sorted in order low to high. Crowding for chromsome x
        # for that score is the difference between the next highest and next
        # lowest score. Total crowding value sums all crowding for all scores
    
        population_size = len(scores[:, 0])
        number_of_scores = len(scores[0, :])
    
        # create crowding matrix of population (row) and score (column)
        crowding_matrix = np.zeros((population_size, number_of_scores))
    
        # normalise scores (ptp is max-min)
        normed_scores = (scores - scores.min(0)) / scores.ptp(0)
    
        # calculate crowding distance for each score in turn
        for col in range(number_of_scores):
            crowding = np.zeros(population_size)
    
            # end points have maximum crowding
            crowding[0] = 1
            crowding[population_size - 1] = 1
    
            # Sort each score (to calculate crowding between adjacent scores)
            sorted_scores = np.sort(normed_scores[:, col])
    
            sorted_scores_index = np.argsort(
                normed_scores[:, col])
    
            # Calculate crowding distance for each individual
            crowding[1:population_size - 1] = \
                (sorted_scores[2:population_size] -
                 sorted_scores[0:population_size - 2])
    
            # resort to orginal order (two steps)
            re_sort_order = np.argsort(sorted_scores_index)
            sorted_crowding = crowding[re_sort_order]
    
            # Record crowding distances
            crowding_matrix[:, col] = sorted_crowding
    
        # Sum crowding distances of each score
        crowding_distances = np.sum(crowding_matrix, axis=1)
    
        return crowding_distances

    def reduce_by_crowding(self, scores, number_to_select):
        # This function selects a number of solutions based on tournament of
        # crowding distances. Two members of the population are picked at
        # random. The one with the higher croding dostance is always picked
        
        population_ids = np.arange(scores.shape[0])
    
        crowding_distances = self.calculate_crowding(scores)
    
        picked_population_ids = np.zeros((number_to_select))
    
        picked_scores = np.zeros((number_to_select, len(scores[0, :])))
    
        for i in range(number_to_select):
    
            population_size = population_ids.shape[0]
    
            fighter1ID = rn.randint(0, population_size - 1)
    
            fighter2ID = rn.randint(0, population_size - 1)
    
            # If fighter # 1 is better
            if crowding_distances[fighter1ID] >= crowding_distances[
                fighter2ID]:
    
                # add solution to picked solutions array
                picked_population_ids[i] = population_ids[
                    fighter1ID]
    
                # Add score to picked scores array
                picked_scores[i, :] = scores[fighter1ID, :]
    
                # remove selected solution from available solutions
                population_ids = np.delete(population_ids, 
                                           (fighter1ID),
                                           axis=0)
    
                scores = np.delete(scores, (fighter1ID), axis=0)
    
                crowding_distances = np.delete(crowding_distances, (fighter1ID),
                                               axis=0)
            else:
                picked_population_ids[i] = population_ids[fighter2ID]
    
                picked_scores[i, :] = scores[fighter2ID, :]
    
                population_ids = np.delete(population_ids, (fighter2ID), axis=0)
    
                scores = np.delete(scores, (fighter2ID), axis=0)
    
                crowding_distances = np.delete(
                    crowding_distances, (fighter2ID), axis=0)
                
        # Convert to integer
        picked_population_ids = np.asarray(picked_population_ids, dtype=int)
        return (picked_population_ids)
    
########## TEST #########

#demo = DemoPareto()
#
#demo.plot_data()
#
#demo.draw_pareto()
#
#demo.draw_knee()