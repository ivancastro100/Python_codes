# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:24:53 2024.

@author: Ivan Castro
"""
from math import log, sqrt, exp
from statistics import NormalDist
import csv
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from datetime import datetime


def black_scholes(exercise_price, interest_rate, maturity_time, option_type,
                  underlying_price, volatility):
    """Get the price of an Eup call or put options using the B-S Formula.

    Parameters
    ----------
    exercise_price: float
        Positive number, exercise price of the contract.
    interest_rate: float
        Positive number, assumed to be annualised, number format between (0,1).
    maturity_time: float
        Positive number, assumed to be annualised.
    option_type: str
        Can be either 'call' or 'put'.
    underlying_price: float
        Positive number, currect price of the underlying asset.
    volatility: float
        Positive number, assumed to be annualised.

    Returns
    -------
    float
        The price of an European call or put

    Raises
    ------
    ValueError
        If interest_rate is not in a format number between (0,1).
    ValueError
        If exercise_price is not positive.
    ValueError
        If underlying_price is not positive.
    ValueError
        If maturity_time is not positive.
    ValueError
        If volatility is not positive.
    ValueError
        If option_type is not 'call' or 'put'

    """
    # Raise errors in case argument conditions do not hold.
    # Change
    if exercise_price < 0:
        raise ValueError(f'{exercise_price = } must be positive.')
    if underlying_price < 0:
        raise ValueError(f'{underlying_price = } must be positive.')
    if maturity_time < 0:
        raise ValueError(f'{maturity_time = } must be positive.')
    if volatility < 0:
        raise ValueError(f'{volatility = } must be positive.')
    if interest_rate > 1 or interest_rate < 0:
        raise ValueError(
            f'{interest_rate = } must be in a number format between (0,1), \
            not a percentage.')
    if option_type != 'call' and option_type != 'put':
        raise ValueError(
            f'{option_type = } can only be either \'call\' or \'put\'')
    # Calculate d1 and d2 for the B-S formula
    d1 = (log(underlying_price/exercise_price) +
          (interest_rate + (((volatility ** 2)/2)) * maturity_time)) \
        / (volatility * sqrt(maturity_time))
    d2 = d1 - (volatility * sqrt(maturity_time))
    if option_type == 'call':
        # Formula for European call
        call = (underlying_price * NormalDist().cdf(d1)) - \
            (exercise_price * exp(-interest_rate * maturity_time) *
             NormalDist().cdf(d2))
        return call
    else:
        # Formula for European put
        put = (exercise_price * exp(-interest_rate * maturity_time) *
               NormalDist().cdf(-d2)) - (underlying_price *
                                         NormalDist().cdf(-d1))
        return put


def find_root(f, a, b, tol=10 ** (-9), max_iter=float('inf')):
    if abs(f(a)) <= tol:
        return a
    elif abs(f(b)) <= tol:
        return b
    elif f(a) * f(b) > 0:
        return None
    else:
        if f(b) < 0:
            xd = b
            xu = a
        else:
            xd = a
            xu = b
        cont = 0
        while cont < max_iter:
            y = (xd * f(xu) - xu * f(xd))/(f(xu) - f(xd))
            if y == xd or y == xu:
                return y
            elif abs(f(y)) <= tol:
                return y
            else:
                if f(y) < 0:
                    xd = y
                else:
                    xu = y
            cont += 1
        return y


def black_scholes_iv(option_price, *, lower_vol=0.0001, upper_vol=100,
                     **k_args):
    return find_root(
        lambda vol: black_scholes(**k_args, volatility=vol) - option_price,
        lower_vol, upper_vol)


def time_series_iv(in_filename, out_filename, plot_filename=None, *,
                   date_field, exercise_price, int_rate_field, iv_field,
                   maturity_field, option_price_field, option_type,
                   underlying_price_field):
    """Write a CSV file with an added field for the implied volatility.

    Parameters
    ----------
    in_filename: str
        Filename of the input CSV.
    out_filename: str
        Filename of the output CSV.
    plot_filename: str
        Filename of the optional plot.
    date_field: str
        The CSV field of the current date that each option is priced at.
    exercise_price: float
        The exercise price of every option in the CSV.
    int_rate_field: str
        The CSV field of the annualised risk-free interest rate of each option.
    iv_field: str
        The CSV field of the implied volatility of each option.
    maturity_field: str
        The CSV field of the maturity time of each option (in days).
    option_price_field: str
        The CSV field of the price of the option.
    option_type: str
        The type of every option in the CSV, either 'call' or 'put'.
    underlying_price_field: str
        The CSV field of the price of the underlying at that date.

    Raises
    ------
    ValueError
        If date_field is not a header of the input file.
    ValueError
        If int_rate_field is not a header of the input file.
    ValueError
        If maturity_field is not a header of the input file.
    ValueError
        If option_price_field is not a header of the input file.
    ValueError
        If underlying_price_field is not a header of the input file.
    ValueError
        If iv_field is a header of the input file.
    """
    # Read the input file.
    with open(in_filename, newline='') as file:
        data = csv.DictReader(file)
        headers = data.fieldnames
        data_list = list(data)
    # Raise errors if the conditions are not met.
    if not (date_field in headers):
        raise ValueError(f'{date_field = } is not a header of the input file.')
    if not (int_rate_field in headers):
        raise ValueError(
            f'{int_rate_field = } is not a header of the input file.')
    if not (maturity_field in headers):
        raise ValueError(
            f'{maturity_field = } is not a header of the input file.')
    if not (option_price_field in headers):
        raise ValueError(
            f'{option_price_field = } is not a header of the input file.')
    if not (underlying_price_field in headers):
        raise ValueError(
            f'{underlying_price_field = } is not a header of the input file.')
    if iv_field in headers:
        raise ValueError(
            f'{iv_field = } should not be header of the input file')
    # Take the values to calculate the imp vol.
    fechas = [i[date_field] for i in data_list]
    op_price = [float(i[option_price_field]) for i in data_list]
    int_rate = [float(i[int_rate_field])/100 for i in data_list]
    maturity = [float(j[maturity_field])/365 for j in data_list]
    underlying = [float(k[underlying_price_field]) for k in data_list]
    # Calculat the imp vol.
    iv = [black_scholes_iv(option_price=op_price[k],
                           exercise_price=exercise_price,
                           interest_rate=int_rate[k],
                           maturity_time=maturity[k],
                           underlying_price=underlying[k],
                           option_type=option_type)*100
          for k in range(len(op_price))]
    # Add the imp vol to the original data.
    for y, z in zip(data_list, iv):
        y[iv_field] = z
    # Write a new CSV with the imp vol in a column at the end.
    with open(out_filename, 'w', newline='') as file2:
        csv_writer = csv.DictWriter(file2, fieldnames=headers+[iv_field])
        csv_writer.writeheader()
        csv_writer.writerows(data_list)
    # Plot in case plot_filename is different of None
    if not (plot_filename is None):
        date_format = '%d%b%Y'
        new_dates = [datetime.strptime(s, date_format) for s in fechas]
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(new_dates, iv)
        ax.set_title('Volatility vs Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Implied volatility (%)')
        ax.xaxis.set_major_locator(MonthLocator())
        fig.savefig(plot_filename)


if __name__ == '__main__':
    # TEST CODE HERE
    name1 = 'option_data.csv'
    name2 = 'new_file1.csv'
    name3 = 'imp_vol'
    name4 = 'volatility_time.png'
    time_series_iv(in_filename=name1, out_filename=name2, plot_filename=name4,
                   date_field='date',
                   exercise_price=3250, int_rate_field='int_rate',
                   iv_field=name3, maturity_field='dtm',
                   option_price_field='mid', option_type='put',
                   underlying_price_field='underlying')
