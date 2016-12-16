import argparse
import json
import numpy as np
import os
from scipy.io import savemat

from scoop import futures

import esnet

###############################################################################################
# The next part needs to be in the global scope, since all workers
# need access to these variables. I got pickling problems when using
# them as arguments in the evaluation function. I couldn't pickle the
# partial function for some reason, even though it should be supported.
############################################################################
# Parse input arguments
############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("data", help="path to data file", type=str)
parser.add_argument("esnconfig", help="path to ESN config file", type=str)
parser.add_argument("savefile", help="path to saved .mat file", type=str)
args = parser.parse_args()

############################################################################
# Read config file
############################################################################
config = json.load(open(args.esnconfig + '.json', 'r'))

############################################################################
# Load data
############################################################################
# If the data is stored in a directory, load the data from there. Otherwise,
# load from the single file and split it.
if os.path.isdir(args.data):
    Xtr, Ytr, _, _, Xte, Yte = esnet.load_from_dir(args.data)

else:
    X, Y = esnet.load_from_text(args.data)

    # Construct training/test sets
    Xtr, Ytr, _, _, Xte, Yte = esnet.generate_datasets(X, Y)

def main():
    # Run in parallel and store result in a numpy array
    Yhat,error,train_states,train_embedding,test_states,test_embedding = esnet.run_from_config_return_states(Xtr, Ytr, Xte, Yte, config)

    savemat(args.savefile, {'train_states':train_states, 'train_embedding':train_embedding, 'test_states':test_states, 'test_embedding':test_embedding})

if __name__ == "__main__":
    main()
