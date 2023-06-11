Compile as follow:

$ make

# To run for rastrigin function with dimension = 10, partical = iterations = 10000
$ ./pso rastrigin 10 10000 -5.12 5.12 10000 4

#change 4 to 8 to 16 for more threads assigned.

# To run for schwefel function with dimension = 20, partical = iterations = 10000
$ ./pso schwefel 20 10000 -500 500 10000 4

#change 4 to 8 to 16 for more threads assigned.
