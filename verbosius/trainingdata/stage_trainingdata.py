import os
import pickle 

import trainingdata.generate_trainingdata as gen_data


def main(batch_dist):

    batch_dist = f"batch_dist_{batch_dist}"
    root = os.path.expanduser('~')
    path = os.path.join(root, "projects/verbosius_data", batch_dist)
    
    n = len(os.listdir(path))/2
    
    for b in range(n):
    
        data = pickle.load(open(f"{path}/data_{b}.pkl", "rb"))

        rm = gen_data.rulemaker(data)



if __name__ == "__main__":

    main(0)