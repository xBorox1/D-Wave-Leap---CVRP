# CVRP approach using quantum computing

Few quantum approaches for CVRPTW problems.

## API Usage

* Prepare test scenario according to Input specification.
* Read the test by function read_full_test or read_test.

## Input specification

You need to prepare 2 files :

* Graph file in simple csv format, named vertex_weights.csv, containing 3 columns as edges: id of first vertes, id of second vertex and weight of edge. Graph is directed.
* Scenario text file. Format is described i 'example_scenario'.

## Running

* Choose a solver.
* Read test by read_test function or read_full_test from input.py
* Run solve function from solver.
