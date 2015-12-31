"""
Chester Holtz

Implementation of relaxed and parallelized Knuth's Five Guess Algorithm[1]
to solve Praetorian's Mastermind Challenge[2]. The Algorithm works as follows:

1. Create the set S of all possible words of constant length k given an alphabet of length n.
2. Start with an initial random guess.
3. Evaluate the score of the guess.
4. Terminate if we have guessed the code.
5. Otherwise, remove from S any code that would not give the same response if it (the guess) were the code.
6. Apply the minimax search technique to find a next guess as follows.
7. Repeat from step 3 until the a score with the number of black pegs equal to the size of the word is obtained.

I relax this algorithm by introducing upper bounds to the number of computations the algorithm preforms on each loop of the program.
Parallelization is achieved with Python's Multiprocessing library against the pruning step. A solution can be found in rare cases for one core, but at least four provides reliable results.
Testing was done on a machine featuring an 8-core Intel i7 processor with 8 MB of Cache and 8 GB of RAM. Additional testing is scheduled on a node server with two processor chips,
each containing 14 cores, each of which has 2 hardware contexts (hyperthreads) - up to 56 threads in parallel.

Future work involves experimenting with alternate datastructures to represent the solution solution space, parallelizing more of the algorithm - ie generating the solution space, and
improving error handling and better printing + io.

[1] Knuth, The Computer as Master Mind (1976)
[2] https://www.praetorian.com/challenges/mastermind/index.html
"""

import sys
import itertools
import multiprocessing
from functools import partial

from random import *
import numpy as np
import interface as interface

""" randomly samples a word from our alphabet """
def random_guess(k,n):
    return sample(k, n)

""" Generates a solution space S by computing every unique permutation of words of length k given an alphabet of length n """
def all_solutions(k,n):
    S_list = []
    S = itertools.permutations(k,n)
    for word in S:
        S_list.append(list(word))
    return S_list

""" Restricts S by initially pruning impossible states given a set of initiall guesses and their scores (hard-coded for level 4) """
def restricted_s(initial_guesses, responses):
    S_restricted = []
    alphabet = range(25)
    tested = []
    for word_guess in initial_guesses:
        tested += word_guess
    nottested = list(set(alphabet) - set(tested))

    index = 0
    count = 0
    allcombs = []
    for guess,response in zip(initial_guesses, responses):
        allcombs.append([])
        # For each subset in a guess, with length equal to the response's "correct count"
        for combination in itertools.combinations(guess, response[0]):
            allcombs[index].append(combination)
        index += 1
        count += response[0]

    # Generate all combinations of untested letters
    if 25 > count:
        allcombs.append([])
        for combination in itertools.combinations(nottested, 6 - count):
            allcombs[index].append(combination)

    # For each combination formed from the results of each guess
    for combination in itertools.product(*allcombs):
        # For each permutation of that combination of the proper length, append to set of combinations
        for permutation in itertools.permutations(sum(combination, ()), 6):
            S_restricted.append(permutation)
    return S_restricted

""" Temporary - I cannot pass ordered arguments with partial """
def filter_wraper(a,b,c,d):
    return filter_matching_result(d,a,b,c)

""" Filter S for words that match a given score when compared with the argument word """
def filter_matching_result(S, w_guess, w_score, max_itr = 10000):
    S_pruned = list(S)
    i = 0
    index = 0
    for s in S:
        i += 1
        if score(w_guess, s) != w_score:
            del S_pruned[index] # optimization over a.remove()
            index -= 1
        if i > max_itr:
            break
        index += 1
    return S_pruned

""" Given two words, compute the score between the two. Return tuple score where
first element is number of elements in w_1, w_2 matching type and location and second element is number of elements of matching type. """
def score(w_1, w_2):
    score = [0,0]
    for a_i, b_i in zip(w_1,w_2):
        if a_i == b_i:
            score[1] += 1
            score[0] += 1
        elif b_i in w_1:
            score[0] += 1
    return score

""" For each element s in S, compute the size of the reduction of S should we choose it as our guess. """
def minimal_eliminated(S, s):
    result_counter = {}
    max_itr = 800 # arbitrary bound
    i = 0
    for s_j in S:
        i += 1
        r_score = tuple(score(s, s_j)) # use resulting score as key
        if r_score not in result_counter.keys():
            result_counter[r_score] = 1
        else:
            result_counter[r_score] += 1
        if i > max_itr:
            return len(S) - max(result_counter.values())
    return len(S) - max(result_counter.values())

""" The best move to choose is the one that results in the smallest solution space """
def best_move(S):
    elim_for_solution = {}
    max_itr = 1000 # arbitrary bound
    i = 0
    for s_i in S:
        i += 1
        elim_for_solution[minimal_eliminated(S, s_i)] = s_i
        if i > max_itr:
            break
    max_elimintated = max(elim_for_solution.keys())
    return elim_for_solution[max_elimintated]

""" interface loop """
def run(CORES):
    iterations = -1
    c_hash = ""
    level = 1
    num_gladiators = 0
    num_weapons = []
    S = [] # solution space
    current_guess = []
    current_score = []

    state = interface.get_new_level(level)
    while True:
        iterations += 1
        for key in state:
            if key == 'message' or key == 'roundsLeft' or iterations == 0: # The occurrence of either of these keywords signifies we have beaten a level of a round
                if key == 'message': # We have beaten a level
                    print state['message']
                    print ''
                    level = level + 1

                state = interface.get_new_level(level)

                for key in state: # End of game
                    if key == 'error': # assume we won
                        print state[key]
                        c_hash = str(interface.get_hash())
                        print("Hash: " + c_hash)
                        f = open('hash.txt', 'a+')
                        f.write("\n")
                        f.write(c_hash)
                        f.close()
                        return 0

                num_weapons = range(state['numWeapons'])
                num_gladiators = state['numGladiators']
                print "LEVEL: " + str(level)
                print "---------------"
                print " A := {" + str(num_weapons)[1:-1] + "}"
                print "|A| := " + str(state["numWeapons"])
                print "|w| := " + str(state["numGladiators"])

                if level != 4:
                    S = all_solutions(num_weapons,state["numGladiators"])
                    print "|S| = " + str(len(S))
                    print "---------------"
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
                    S = restricted_s(guesses, responses)
                    print "|S| = " + str(len(S))
                    print "--------"

                current_guess = random_guess(num_weapons,num_gladiators)
                print "Initial Guess: " + str(current_guess)
                state = interface.submit_guess(level,current_guess)
                break
        current_score = state['response']

        # multiprocessing - https://www.praetorian.com/blog/multi-core-and-distributed-programming-in-python
        pool = multiprocessing.Pool(processes=CORES)
        divide_S = [list(S)[i::CORES] for i in xrange(CORES)]
        p_filter = partial(filter_wraper, current_guess,current_score,20000)
        S = pool.map_async(p_filter,divide_S)
        S = S.get(10)
        pool.close()
        pool.join()
        S = [s for sublist in S for s in sublist] # flatten list

        print "|S| reduced to " + str(len(S))
        current_guess = best_move(S)
        # print "current_guess: " + str(current_guess)
        state = interface.submit_guess(level,current_guess)

def main():
    CORES = 2
    # takes single optional core argument. Default is 2.
    if len(sys.argv) > 0:
        if '-c' in sys.argv:
            CORES = int(sys.argv[2])
    interface.reset_game()
    run(CORES) # change
    return 0

if __name__ == '__main__':
    main()
