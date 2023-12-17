# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:35:56 2023

@author: sarah
"""

# main.py

from beehive import GeneticAlgorithm, plot_genealogy,hierarchy_pos,_hierarchy_pos
import numpy as np 
import pandas as pd # Importez d'autres modules si nécessaire
import networkx as nx
import matplotlib.pyplot as plt

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisez vos données, par exemple :
    population_size = 100
    df = pd.read_csv('abeilles.csv')
    # Ajouter la position de la ruche
    position_ruche = pd.DataFrame({'x': [500], 'y': [500]})
    df = pd.concat([position_ruche, df], ignore_index=True)

    # Créez une instance de la classe GeneticAlgorithm
    ga_instance = GeneticAlgorithm(population_size, df, method='top_parents', crossover_method='crossover', mutation_method='inversion', seed=42)

    # Appelez les méthodes de l'instance selon vos besoins
    best_distance, best_parcours, mean_iteration_time, best_parcours_info = ga_instance.run_genetic_algorithm(generations=60, mutation_rate=0.04, show_plot=True)
    
    parents_ids = ga_instance.get_parents_ids()
    best_abeille_id = best_parcours_info[0]
    print(best_abeille_id)
    # Appelez la fonction plot_genealogy
    plot_genealogy(parents_ids, best_abeille_id,generations=5)

    # Affichez le graphique
    plt.show()