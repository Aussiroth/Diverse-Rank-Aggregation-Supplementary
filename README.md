# Diverse Rank Aggregation

Supplementary material and code for the paper on 'Rank Aggregation - beyond a single consensus'.

The code has been tested to work on Python 3.13, 3.14, and on Windows 11 and Ubuntu 24.04.

## Necessary python packages

Note that the modules in this repo utilizes a number of standard packages used in data science, and exact solvers:
-numpy, scipy, pandas, networkx, ortools, cvxpy.

The exact packages used are in `requirements.txt`. Note that highspy on versions newer than 1.11 can cause conflicts with ortools on some systems. In this case, reinstall highspy version 1.11 with ```python -m pip install --force-reinstall highspy==1.11``` (or the corresponding command on your system).

## Datasets
Datasets are found in the individual folders Movielens, country and university. The data is already preprocessed for input to our code. You can verify the datasets indeed correspond to the original using the dataset citations in our paper. Country and university can be found on the preflib website; we use the already preprocessed full ranking inputs.

### Custom Datasets
To test our algorithms with other datasets, note that the expected form of the input file is n lines, each with the numbers 0 ... d-1 in order (for d candidates). Each line represents one ranking over the d candidates. Any other input formats will lead to an error eventually being thrown.

### Synthetic Dataset

To generate synthetic dataset for the scalability experiments, see `generate_rankings.py` in the `synthetic` folder. It writes the instance to standard output.

Examples: 

- ```python generate_rankings.py -n 1000 -d 500 --model uniform``` 
will create an instance with 1000 rankings over 500 candidates using a uniform distribution, and can be directly used in our data. The seed is fixed to 0 to reproduce the inputs used; otherwise, adding the command line --seed [SEED] allows you to modify it.
- ```python generate_rankings.py -n 1000 -d 500 --model mallows --alpha 2``` will create an instance using the mallows distribution instead, where rankings are sampled proportional to e^(-alpha * d) where d is the distance of the ranking from the identity ranking.

## Running the code
`main.py` is the main entrance to run all our algorithms on a single instance. It takes in `k`, the number of output rankings as a command line parameter, and takes from standard input the path to the file to read as input data.

Running `python main.py k` where `k` is an integer will run each of our algorithms with the goal of finding `k` aggregate rankings. The code also expects to take in from standard input the path to the file to read as input.

You can try running `python main.py 3 < input.txt` as an example. This has our algorithms find 3 aggregate rankings, and uses the university (2015) dataset. The writing to the standard out consists of multiple data points, comma separated.

Note that CP-SAT has an aggressive timeout by design, as those can take hours to complete otherwise. To run it with a large timeout, change it manually (see the top of `cp_sat.py`).