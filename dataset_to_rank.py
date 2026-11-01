import pandas as pd
import numpy as np
from scipy.stats import rankdata

def getDataset(dataset_path):
    # Read directly from the uploaded text file
    file_name = dataset_path
    full_rankings = []
    id_mapping = {}
    id = 1

    with open(file_name, 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split the row by spaces
            ranking = line.strip().split(' ')
            
            # Create the dictionary: {input candidate id: rank}
            # Index 0 gets Rank 1, so we add 1 to the enumeration index
            user_ranking = {}
            for rank, movie in enumerate(ranking):
                if movie not in id_mapping:
                    id_mapping[movie] = id
                    id += 1
                user_ranking[id_mapping[movie]] = rank + 1

            full_rankings.append(user_ranking)

    return full_rankings