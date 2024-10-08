# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:44:06 2024.

@author: Ivan Castro
"""
from math import sqrt, exp
from random import gauss


def vasicek_sim(initial, final_time, sim_path_len, num_paths=None, *, a, b,
                sigma, allow_neg=True):  # check none, k arg
    """Funtion generates paths Vasicek rate model using Monte-Carlo simulation.

    Parameters
    ----------
    initial: float or list
        A number or a list of numbers with the current rate at the end.
    final_time: float
        A positive number, last point in time.
    sim_path_len: int
        A positive integer, num of points after the initial data.
    num_paths: int or None
        Number of simulated paths, if None simulate a single path
    a, b, sigma: float
        Positive numbers, arguments of the Vasicek model.
    allow_neg: bool
        If False, the fuction stop simulation if find a negative number.

    Returns
    -------
    rate_final: list
    Return a list with simulated paths of the interest rate.
    """
    delta = final_time / sim_path_len
    rate_final = []
    if num_paths is None:  # In case num_paths is None, retunr single path
        if type(initial) is list:  # If initial is a list we use extend
            rate_final.extend(initial)
        else:
            rate_final.append(initial)  # If initial is a number use append
        for j in range(sim_path_len):
            normal = gauss(0, 1)  # Standar Normal random variable
            # Compute the new value of the interest rate using the previous
            r = (exp(-a*delta)*rate_final[-1])+(b*(1-exp(-a*delta))) + \
                sigma*sqrt((1-exp(-2*a*delta))/(2*a))*normal
            if r < 0 and allow_neg is False:
                break    # In case allow_neg is False stop
            rate_final.append(r)
    else:      # In case num_paths is an int positive, retunr a list of lists
        for i in range(num_paths):
            rate = []
            if type(initial) is list:
                rate.extend(initial)
            else:
                rate.append(initial)
            for j in range(sim_path_len):
                normal = gauss(0, 1)  # Standar Normal random variable
                # Compute the new value of the interest rate using the previous
                r = (exp(-a*delta)*rate[-1])+(b*(1-exp(-a*delta))) + \
                    sigma*sqrt((1-exp(-2*a*delta))/(2*a))*normal
                if r < 0 and allow_neg is False:
                    break   # In case allow_neg is False stop
                rate.append(r)
            rate_final.append(rate)
    return rate_final


"""  Testing """
if __name__ == '__main__':
    aa = vasicek_sim(initial=0.01095129, final_time=10,
                     sim_path_len=40, num_paths=None, a=0.09, b=0.03, sigma=0.02)
    print(aa)
