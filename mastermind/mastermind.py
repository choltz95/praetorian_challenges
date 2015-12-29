"""
Chester Holtz 

Implementation of relaxed and parallelized Knuth's Five Guess Algorithm [1]
to solve Praetorian's Mastermind Challenge [2]. The Algorithm works as follows:

1. Create the set S of all possible words given an alphabet.
2. Start with an initial random guess.
3. Evaluate the score of the guess.
4. Terminate if we have guessed the code.
5. Otherwise, remove from S any code that would not give the same response if it (the guess) were the code.
6. Apply the minimax search technique to find a next guess as follows.
7. Repeat from step 3 until the a score with the number of black pegs equal to the size of the word is obtained.

I relax this algorithm by introducing upper bounds to the number of computations the algorithm preforms on each loop of the program. 
Parallelization is achieved with Python's Multiprocessing library against the pruning step. A solution can be found in rare cases for one core, but at least four provides reliable results.
Testing was done on a machine featuring an 8-core Intel i7 processor with 8 MB of Cache and 8 GB of RAM. Additional testing is scheduled on a node server with two processor chips, 
each containing 14 cores, each of which has 2 hardware contexts (hyperthreads) - up to 56 threads in parallel. Writeup to come

Future work involves experimenting with alternate datastructures to represent the solution solution space, parallelizing more of the algorithm - ie generating the solution space, and 
improving error handling and io.

[1] Knuth, The Computer as Master Mind (1976)
[2] https://www.praetorian.com/challenges/mastermind/index.html
"""

import itertools
import multiprocessing
from functools import partial

from random import *
import numpy as np
import interface as interface

""" randomly samples a word from our alphabet """
def random_guess(num_weapons,numGladiators):
    return sample(num_weapons, numGladiators)

""" Generates a solution space S by computing every unique permutation of words of length k given an alphabet of length n """
def all_solutions(num_weapons,numGladiators):
    x = []
    perms = itertools.permutations(num_weapons,numGladiators)
    for perm in perms:
        x.append(list(perm))
    return x

""" Restricts S by initially pruning impossible states given a set of initiall guesses and their scores """
def generate_possible(initial_guesses, responses=None):
    # currently hardcoded for level 4
    possible = [] 
    weaponlist = range(25)
    alltested = []
    for each in initial_guesses:
        alltested += each
    #alltested is now every element that has been tested
    nottested = list( set(weaponlist) - set(alltested) )

    index = 0
    count = 0
    allcombs = []
    for guess,response in zip(initial_guesses, responses):
        allcombs.append([])
        #For each subset in that guess, with length equal to the response's "correct count"
        for combination in itertools.combinations(guess, response[0]):
            allcombs[index].append(combination)
        index += 1
        count += response[0]

    #Handling the not tested group of weapons
    if 25 > count:
        allcombs.append([])
        for combin in itertools.combinations(nottested, 6 - count):
            allcombs[index].append(combin)

    #For each combination formed from the results of each guess
    for combination in itertools.product(*allcombs):
        #For each permutation of that combination of the proper length
        for perm in itertools.permutations(sum(combination, ()), 6):
            possible.append(perm)
    return possible

""" Temporary - I cannot pass ordered arguments with partial """
def filter_wraper(a,b,c,d):
    return filter_matching_result(d,a,b,c)

""" Filter S for solutions that match a given score when compared with a word """
def filter_matching_result(solution_space, guess, result, max_itr = 10000):
    pruned_space = list(solution_space)
    itr = 0
    index = 0
    for solution in solution_space:
        itr += 1
        if score(guess, solution) != result:
            del pruned_space[index] # optimization over a.remove()
            index -= 1
        if itr > max_itr:
            break
        index += 1
    return pruned_space

""" Given two words, compute the score between the two """
def score(actual, guess):
    result = [0,0]
    for x, y in zip(actual,guess):
        if x == y:
            result[1] += 1
            result[0] += 1
        elif y in actual:
            result[0] +=1
    return result

""" For each element of S, compute the size of the reduction from S should we choose it as our guess. """
def minimal_eliminated(solution_space, solution):
    result_counter = {}
    max_itr = 800
    itr = 0
    for option in solution_space:
        itr += 1
        result = tuple(score(solution, option)) # use score as key
        if result not in result_counter.keys():
            result_counter[result] = 1
        else:
            result_counter[result] += 1
        if itr > max_itr:
            return len(solution_space) - max(result_counter.values())
    return len(solution_space) - max(result_counter.values())

""" The best move to choose ise the one that results in the smallest solution space """
def best_move(solution_space):
    elim_for_solution = {}
    max_itr = 1000
    itr = 0

    for solution in solution_space:
        itr += 1
        elim_for_solution[minimal_eliminated(solution_space, solution)] = solution
        if itr > max_itr:
            break
    max_elimintated = max(elim_for_solution.keys())
    return elim_for_solution[max_elimintated]

""" interface loop """
def run(Cores):
    # initialize
    iterations = -1
    c_hash = ""
    level = 1
    num_weapons = []
    num_gladiators = 0
    solution_space = []
    current_guess = []
    current_score = []

    state = interface.get_new_level(level)
    while True:
        iterations += 1
        if interface.get_hash() is not None: # End of game
            c_hash = str(interface.get_hash())
            print("Hash: " + c_hash)
            f = open('hash.txt', 'a+')
            f.write(c_hash)
            f.close()
            return 0
        else:
            for key in state:
                if key == 'message' or key == 'roundsLeft' or iterations == 0: # The occurrence of either of these keywords signifies we have beaten a level of a round
                    if key == 'message': # We have beaten a level
                        print state['message']
                        level = level + 1
                    print "*****LEVEL: " + str(level)

                    state = interface.get_new_level(level)
                    num_weapons = range(state['numWeapons'])
                    num_gladiators = state['numGladiators']
                    print state

                    if level != 4:
                        solution_space = all_solutions(num_weapons,state["numGladiators"])
                        print "solution_space computed"
                    elif level == 4:
                        guesses = []
                        responses = []
                        counter = state['numWeapons']
                        greatest = 0
                        while counter > 6:
                            guess = []
                            for x in range(greatest, greatest+6):
                                guess.append(x)
                            counter -= len(guess)
                            greatest = max(guess) + 1

                            guesses.append(guess)
                            state = interface.submit_guess(level, guess)
                            feedback = state['response']
                            responses.append(feedback)
                        solution_space = generate_possible(guesses, responses)

                    current_guess = random_guess(num_weapons,num_gladiators)
                    print current_guess
                    state = interface.submit_guess(level,current_guess)
                    break
            current_score = state['response']
            # multiprocessing - https://www.praetorian.com/blog/multi-core-and-distributed-programming-in-python
            pool = multiprocessing.Pool(processes=Cores)
            divide_S = [list(solution_space)[i::Cores] for i in xrange(Cores)]
            p_filter = partial(filter_wraper, current_guess,current_score,20000)
            solution_space = pool.map_async(p_filter,divide_S)
            solution_space = solution_space.get(10)
            pool.close()
            pool.join()
            solution_space = [ent for sublist in solution_space for ent in sublist]
            print "soln space filtered: " + str(len(solution_space))

            current_guess = best_move(solution_space)
            print current_guess
            state = interface.submit_guess(level,current_guess)

def main():
    interface.reset_game()
    run(10) # change
    return 0

if __name__ == '__main__':
    main()
