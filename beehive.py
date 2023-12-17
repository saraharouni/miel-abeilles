# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:36:00 2023

@author: sarah
"""

import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import time
from typing import Optional, List, Tuple

import networkx as nx

def hierarchy_pos(G, root=None, width=1., vert_gap=0.3, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)

    if len(children) != 0:
        dx = width / 2
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap * 2, xcenter=nextx, pos=pos,
                                 parent=root, parsed=parsed)

    return pos


def plot_genealogy(parents_ids, target_id, generations=5):
    G = nx.DiGraph()

    def add_parents_to_graph(graph, current_id, generations_left):
        if generations_left <= 0:
            return

        parents = parents_ids.get(current_id)
        if parents:
            parent1, parent2 = parents
            graph.add_edge(current_id, parent1)
            graph.add_edge(current_id, parent2)

            # Récursivement ajouter les parents
            add_parents_to_graph(graph, parent1, generations_left - 1)
            add_parents_to_graph(graph, parent2, generations_left - 1)

    add_parents_to_graph(G, target_id, generations)

    pos = hierarchy_pos(G, target_id, vert_gap=0.8)
    plt.figure(figsize=(12, 9))  # Ajustez les valeurs de largeur et hauteur selon vos besoins
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=600, node_color='skyblue', font_size=8, arrowsize=15)
    plt.title(f'Généalogie de l\'abeille avec ID {target_id} sur {generations} générations')
    plt.show()

class GeneticAlgorithm:
    unique_id = 0

    def __init__(self, population_size, df, method='top_parents', crossover_method='crossover', mutation_method='inversion', seed=42):
        self.population_size = population_size
        self.df = df
        self.seed = seed
        self.abeille_counter = 0  # Compteur d'abeilles pour générer des IDs uniques
        self.parcours_abeilles = self.initialize_population()
        self.dist_mat = self.calculate_distance_matrix()
        self.method = method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.parents_ids = {}  # Dictionnaire pour répertorier les IDs des parents

    @classmethod
    def generate_unique_id(cls):
        cls.unique_id += 1
        return cls.unique_id

    def initialize_population(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        parcours_abeilles = []
        for _ in range(self.population_size):
            parcours = list(range(1, len(self.df)))  # Commence à partir de 1
            np.random.shuffle(parcours)
            if 0 not in parcours:
                parcours = [0] + parcours  # Ajoute 0 au début du parcours
                abeille_id = self.generate_unique_id()
                parcours_abeilles.append((abeille_id, self.ajuster_parcours(parcours)))
            else:
                abeille_id = self.generate_unique_id()
                parcours_abeilles.append((abeille_id, self.ajuster_parcours(parcours)))
        return parcours_abeilles
    
    def get_parents_ids(self):
        return self.parents_ids
    
    def ajuster_parcours(self, parcours):
        position_ruche = parcours.index(0)
        return parcours[position_ruche:] + parcours[:position_ruche]

    def total_distance(self, order):
        distances = [self.dist_mat[order[i], order[(i + 1) % len(order)]] for i in range(len(order))]
        return sum(distances)

    def calculate_distance_matrix(self):
        A = np.array(self.df)
        B = A.copy()
        return distance_matrix(A, B, p=1)

    def crossover_two_points(self, parent1, parent2):
        point1, point2 = 1, 49  # Indices de coupure arbitraires
        child1 = [0] + parent1[point1:point2]
        child2 = [0] + parent2[point1:point2]

        remaining_parent1 = [gene for gene in parent2 if gene not in child1]
        remaining_parent2 = [gene for gene in parent1 if gene not in child2]

        child1 += remaining_parent1
        child2 += remaining_parent2

        return child1, child2

    def crossover(self, parent1, parent2):
        start = np.random.randint(2, len(parent1))  # Démarre à partir de 2 pour exclure la ruche
        end = np.random.randint(start, len(parent1))
        child = [0] + [None] * (len(parent1) - 1)
        for i in range(start, end):
            child[i] = parent1[i]
        idx = 1
        for i in range(len(parent2)):
            if parent2[i] not in child:
                while child[idx] is not None:
                    idx += 1
                child[idx] = parent2[i]
        return child

    def top_parents_selection(self, num_parents=50):
        sorted_population = sorted(self.parcours_abeilles, key=lambda x: self.total_distance(x[1]))
        return sorted_population[:num_parents]

    def tournament_selection(self, k=5):
        indices = np.random.choice(len(self.parcours_abeilles), size=k, replace=False)
        candidates = [self.parcours_abeilles[i] for i in indices]
        winner = min(candidates, key=lambda x: self.total_distance(x[1]))
        return winner

    def roulette_wheel_selection(self):
        top_50 = self.parcours_abeilles[:50]
        total_fitness = sum([1 / self.total_distance(parcours[1]) for parcours in top_50])
        probabilities = [(1 / self.total_distance(parcours[1])) / total_fitness if total_fitness != 0 else 1/len(top_50) for parcours in top_50]
        selected_index = np.random.choice(range(len(top_50)), p=probabilities)
        return top_50[selected_index].copy()

    def mutate_swap(self, order):
        idx1, idx2 = np.random.choice(range(1, len(order)), 2, replace=False)  # Commence à partir de 1
        order[idx1], order[idx2] = order[idx2], order[idx1]
        return order

    def mutate_inversion(self, order):
        start, end = np.random.choice(range(1, len(order)), 2, replace=False)  # Commence à partir de 1
        order[start:end + 1] = order[start:end + 1][::-1]
        return order

    def mutate_insertion(self, order):
        idx1, idx2 = np.random.choice(range(1, len(order)), 2, replace=False)  # Commence à partir de 1
        city_to_insert = order[idx1]
        order.pop(idx1)
        order.insert(idx2, city_to_insert)
        return order

    def ensure_zero_first(self, order):
        zero_index = order.index(0)
        order[0], order[zero_index] = order[zero_index], order[0]
        return order
    
            
    def run_genetic_algorithm(self, generations=800, mutation_rate=0.04,show_plot=False):
        fitness_history = []
        iteration_times = []

        best_fitness = float('inf')  # Initialise à l'infini
        stagnation_counter = 0

        for generation in range(generations):
            start_time = time.time()
            self.parcours_abeilles = sorted(self.parcours_abeilles, key=lambda x: self.total_distance(x[1]))
            new_parcours_abeilles = []

            for i in range(len(self.parcours_abeilles) // 2):
                parent1 = self.parcours_abeilles[i]
                parent2 = self.parcours_abeilles[i + 1]

                if self.method == "tournament":
                    parent1 = self.tournament_selection()
                    parent2 = self.tournament_selection()
                elif self.method == "roulette":
                    parent1 = self.roulette_wheel_selection()
                    parent2 = self.roulette_wheel_selection()
                elif self.method == "top_parents":
                    parents = self.top_parents_selection(2)
                    parent1, parent2 = parents[0], parents[1]
                else:
                    raise ValueError("Méthode de sélection des parents non valide")

                if self.crossover_method == "crossover":
                    child1 = (self.abeille_counter, self.crossover(parent1[1], parent2[1]))
                    child2 = (self.abeille_counter, self.crossover(parent2[1], parent1[1]))
                elif self.crossover_method == "crossover_two_points":
                    child1, child2 = self.crossover_two_points(parent1[1], parent2[1])
                else:
                    raise ValueError("Méthode de croisement non valide")

                if np.random.rand() < mutation_rate:
                    if self.mutation_method == "swap":
                        child1 = (self.abeille_counter, self.mutate_swap(child1[1]))
                    elif self.mutation_method == "inversion":
                        child1 = (self.abeille_counter, self.mutate_inversion(child1[1]))
                    elif self.mutation_method == "insertion":
                        child1 = (self.abeille_counter, self.mutate_insertion(child1[1]))
                    else:
                        raise ValueError("Méthode de mutation non valide")

                if np.random.rand() < mutation_rate:
                    if self.mutation_method == "swap":
                        child2 = (self.abeille_counter, self.mutate_swap(child2[1]))
                    elif self.mutation_method == "inversion":
                        child2 = (self.abeille_counter, self.mutate_inversion(child2[1]))
                    elif self.mutation_method == "insertion":
                        child2 = (self.abeille_counter, self.mutate_insertion(child2[1]))
                    else:
                        raise ValueError("Méthode de mutation non valide")

                new_parcours_abeilles.extend([child1, child2])

                # Répertorie les IDs des parents
                self.parents_ids[child1[0]] = (parent1[0], parent2[0])
                self.parents_ids[child2[0]] = (parent1[0], parent2[0])

                # Incrémente le compteur d'abeilles
                self.abeille_counter += 1

            self.parcours_abeilles = new_parcours_abeilles

            best_parcours_info = min(self.parcours_abeilles, key=lambda x: self.total_distance(x[1]))
            best_parcours_id, best_parcours = best_parcours_info[0], best_parcours_info[1]
            best_distance = self.total_distance(best_parcours)
            fitness_history.append(best_distance)

            # Vérifie la stagnation
            if best_distance >= best_fitness:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_fitness = best_distance

            # Applique les mutations seulement en cas de stagnation
            if stagnation_counter > 50:
                for abeille_info in self.parcours_abeilles:
                    abeille_id, parcours = abeille_info[0], abeille_info[1]
                    if np.random.rand() < mutation_rate:
                        mutated_parcours = None
                        if self.mutation_method == "swap":
                            mutated_parcours = self.mutate_swap(parcours)
                        elif self.mutation_method == "inversion":
                            mutated_parcours = self.mutate_inversion(parcours)
                        elif self.mutation_method == "insertion":
                            mutated_parcours = self.mutate_insertion(parcours)

                        if mutated_parcours is not None:
                            self.parcours_abeilles.append((self.abeille_counter, mutated_parcours))
                            self.abeille_counter += 1

            end_time = time.time()
            iteration_time = end_time - start_time
            iteration_times.append(iteration_time)

        if show_plot:
            plt.plot(fitness_history)
            plt.title(f'Evolution du fitness par génération\nSelection: {self.method}, Crossover: {self.crossover_method}, Mutation: {self.mutation_method}, Mutation Rate: {mutation_rate}')
            plt.xlabel('Génération')
            plt.ylabel('Fitness Score')
            plt.show()

        print("Meilleur parcours trouvé :", best_parcours)
        print("Distance totale :", best_distance)
        print("Temps moyen par génération :", np.mean(iteration_times))

        return best_distance, best_parcours, np.mean(iteration_times),best_parcours_info  # Retourne également
