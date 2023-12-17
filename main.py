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


if __name__ == "__main__":
    population_size = 100
    df = pd.read_csv('abeilles.csv')
    position_ruche = pd.DataFrame({'x': [500], 'y': [500]})
    df = pd.concat([position_ruche, df], ignore_index=True)

    # Création d'une instance de la classe GeneticAlgorithm
    ga_instance = GeneticAlgorithm(population_size, df=df,method='tournament', crossover_method='crossover', mutation_method='inversion', seed=42)

    # Appel des méthodes de l'instance
    best_distance, best_parcours, mean_iteration_time, best_parcours_info = ga_instance.run_genetic_algorithm(generations=600, mutation_rate=0.07, show_plot=True)
    
    parents_ids = ga_instance.get_parents_ids()
    best_abeille_id = best_parcours_info[0]
    print(best_abeille_id)
    # Création et affichage de l'arbre généalogique
    plot_genealogy(parents_ids, best_abeille_id,generations=5)
    plt.show()