import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ############################
# ###### First Meeting #######
# ############################

start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2024, 8, 23)

# Yahoo finance API
# Final stock allocation
stocks = ['AAPL', 'MSFT', 'JNJ', 'PFE', 'JPM', 'BAC', 'PG', 'KO', 'XOM', 'CVX']
m = len(stocks)
data = yf.download(stocks, start=start, end=end, interval='1wk')  # Weekly
close = data['Adj Close']
returns = close.pct_change().fillna(0)
growth_fac = 1 + returns
log_returns = np.log(growth_fac)
composed_ret = growth_fac.apply(np.prod, axis=0)  # Composed returns


# ############## Task 1.5 ##################
# Rebalancing Portfolio (Equation 1)
theta = np.full(m, 1/(m + 1))  # Weights
rebalance_port = (1 + (returns @ theta)).cumprod()
log_rebalance = np.log(rebalance_port)

# Buy and hold Portfolio (Equation 2)
initial = close.iloc[0]
ratio = close / initial
weighted_ratio = ratio.dot(theta)  # Weights
BnH = (1 - theta.sum()) + weighted_ratio
log_BH = np.log(BnH)

# Market Portfolio
market_values = close.sum(axis=1)
market_returns = market_values.pct_change().fillna(0)
market_port = (1 + market_returns).cumprod()
log_market = np.log(market_port)

# Plot the log values of both portfolios and Market Portfolio
plt.figure(figsize=(14, 7), dpi=300)
plt.plot(log_rebalance, label='Log of Rebalancing Portfolio',
         color='mediumblue')
plt.plot(log_BH, label='Log of Buy and Hold Portfolio', color='green')
plt.plot(log_market, label='Log of Market Portfolio',
         linestyle='--', color='red')
plt.title('Portfolio Values Over Time')
plt.xlabel('Date')
plt.ylabel('Log of Vt')
plt.legend()
plt.savefig('1_bnh_reb.png')
plt.show()


# AVDD and MDD functions
# ###############################################################
def AVDD(cumulative_returns):
    peak = cumulative_returns.expanding(min_periods=1).max()
    dd = (cumulative_returns / peak) - 1
    return dd.mean()


# ###############################################################
def MDD(cumulative_returns):
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()
# ##############################################################

# #################################
# ######## Second Meeting #########
# #################################


# ################# Task 2.4 #################
Z = len(rebalance_port)  # Total periods
time = np.arange(1, Z+1)

# A1 - Portfolio of Portfolio
# Calculate and plot PoP for Rebalancing, Buy-n-hold and Market
Wt_rebalance = (1/Z) * rebalance_port.cumsum() + \
    rebalance_port * ((Z - time) / Z)
Wt_bnh = (1/Z) * BnH.cumsum() + BnH * ((Z - time) / Z)
Wt_market = ((1/Z) * market_port.cumsum() + market_port * ((Z - time) / Z))

log_Wt_rebalance = np.log(Wt_rebalance)
log_Wt_bnh = np.log(Wt_bnh)
log_Wt_market = np.log(Wt_market)

# Plot
plt.figure(figsize=(14, 7), dpi=300)
plt.plot(log_Wt_rebalance, label='Log of PoP Rebalancing Portfolio',
         color='mediumblue')
plt.plot(log_Wt_bnh, label='Log of PoP Buy and Hold Portfolio', color='green')
plt.plot(log_Wt_market, label='Log of PoP Market Portfolio',
         linestyle='--', color='red')
plt.title('Portfolio of Portfolios (PoP) Values Over Time')
plt.xlabel('Date')
plt.ylabel('Log of PoP')
plt.legend()
plt.savefig('2_pop_bnh_reb.png')
plt.show()


# B1 - CK Allocation for Vt
# #######################################################################
def ck_allocation(data, alpha, reg=1e-4):
    cum_returns = data.cumsum()
    raw_covariance = cum_returns.T @ cum_returns
    # Ridge regularization, stabilize the inversion of the cov matrix
    reg_covariance = raw_covariance + np.eye(raw_covariance.shape[0]) * reg
    inv_reg_covariance = np.linalg.inv(reg_covariance)
    raw_allocation = alpha * (inv_reg_covariance @ cum_returns.iloc[-1].values)
    raw_allocation[raw_allocation < 0] = 0
    if raw_allocation.sum() > 1:
        return raw_allocation / raw_allocation.sum()
    else:
        return raw_allocation
# #######################################################################


alphas = [10, 4, 1, 0.5, 0.10]  # Corrected alpha values
V = {alpha: np.ones(Z) for alpha in alphas}

for t in range(1, Z):
    for a in alphas:
        allocation1 = ck_allocation(returns.iloc[:t], a)
        portfolio_return = (allocation1 * returns.iloc[t]).sum()
        V[a][t] = V[a][t-1] * (1 + portfolio_return)

V_df = pd.DataFrame(V, index=returns.index)

plt.figure(figsize=(14, 7), dpi=300)
for a in alphas:
    plt.plot(np.log(V_df[a]), label=f'Log of Portfolio for α={a}')
plt.plot(log_market, label='Log of Market Portfolio',
         linestyle='--', color='red')
plt.title('Portfolio Values for selected α-CK Allocations Over Time')
plt.xlabel('Date')
plt.ylabel('Log of Vt')
plt.legend()
plt.savefig('3_ck.png')
plt.show()


# ################################################################
# Xt / X0 plot
plt.figure(figsize=(14, 7), dpi=300)
log_ratio = np.log(close / initial)
for stock in stocks:
    plt.plot(close.index, log_ratio[stock], label=stock)
plt.title('Stock Prices Relative to Initial Values')
plt.xlabel('Date')
plt.ylabel('Log of Vt')
plt.legend()
plt.savefig('4_stocks.png')
plt.show()


# 3 Theta-plots ##################################################
theta_s = {alpha: [] for alpha in alphas}
cash_s = {alpha: [] for alpha in alphas}
for alpha in alphas:
    initial_theta = np.ones(m) / (m + 1)
    theta_s[alpha].append(initial_theta)
    cash_s[alpha].append(1 - initial_theta.sum())

    for t in range(1, Z):
        allocation = ck_allocation(returns.iloc[:t], alpha)
        allocation = np.clip(allocation, 0, None)
        if allocation.sum() > 1:
            allocation /= allocation.sum()
        theta_s[alpha].append(allocation)
        cash_s[alpha].append(1 - allocation.sum())

    theta_df = pd.DataFrame(theta_s[alpha], index=returns.index)
    cash_df = pd.Series(cash_s[alpha], index=returns.index)

'''
    plt.figure(figsize=(14, 7))
    for i in range(m):
        plt.plot(theta_df.index, theta_df.iloc[:, i], label=stocks[i])
    plt.plot(cash_df.index, cash_df, label='Cash', linestyle='--')
    plt.title(f'Theta(t) and Cash Position for α={alpha}')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.legend()
    plt.show()
'''

fig, axes = plt.subplots(3, 2, figsize=(14, 8), dpi=300)
fig.suptitle('Theta-plots for Different Alpha Values', fontsize=16)
# Flatten the axes array for easier indexing
axes = axes.flatten()

for idx, alpha in enumerate(alphas):
    ax = axes[idx]
    theta_df = pd.DataFrame(theta_s[alpha], index=returns.index)
    cash_df = pd.Series(cash_s[alpha], index=returns.index)

    for i in range(m):
        ax.plot(theta_df.index, theta_df.iloc[:, i], label=stocks[i])
    ax.plot(cash_df.index, cash_df, label='Cash', linestyle='--')

    ax.set_title(f'Theta(t) and Cash Position for α={alpha}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1

# Remove any unused subplots
for i in range(len(alphas), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('5_theta.png')
plt.show()


# ################################################################
# 4 Plot log Wt (PoP) for the 5 time series
Wt = {alpha: np.zeros(Z) for alpha in alphas}
for t in range(Z):
    for alpha in alphas:
        Wt[alpha][t] = (1/Z) * V_df[alpha].iloc[:t+1].sum() + \
            V_df[alpha].iloc[t] * ((Z - t) / Z)

Wt_df = pd.DataFrame(Wt, index=returns.index)

plt.figure(figsize=(14, 7), dpi=300)
for alpha in alphas:
    plt.plot(np.log(Wt_df[alpha]), label=f'Log PoP for α={alpha}')
plt.plot(log_Wt_market, label='Log PoP Market Portfolio',
         linestyle='--', color='red')
plt.title('Portfolio of Portfolios for selected α-CK Allocations Over Time')
plt.xlabel('Date')
plt.ylabel('Log of PoP')
plt.legend()
plt.savefig('6_pop_ck.png')
plt.show()


# #################################
# ######## Third Meeting ##########
# #################################

# ############# Task 3.3.1: Learning alpha part 1 #############
# Aggregate Portfolio

# V1 - Buy and Hold Portfolio
V1 = BnH

# V2 - CK-Allocation Portfolio with alpha = 10
alp = 10  # alpha value
V2 = np.ones(Z)
theta_agg = []

for t in range(1, Z):
    allocation3 = ck_allocation(returns.iloc[:t], alp)
    portfolio_return3 = (allocation3 * returns.iloc[t]).sum()
    V2[t] = V2[t-1] * (1 + portfolio_return3)
    # Calculate theta_agg
    theta_agg_t = (V1[t-1] * (1/(m + 1)) + V2[t-1] *
                   allocation3) / (V1[t-1] + V2[t-1])
    theta_agg.append(theta_agg_t)
# np.zeros(m)
theta_agg = np.array(theta_agg)
V2_df = pd.DataFrame(V2, index=returns.index)

# Aggregate Buy and Hold and CK-allocation portfolio (alpha=10)
V_agg2 = 0.5 * (V1 + V2)
V_agg2_df = pd.DataFrame(V_agg2, index=returns.index)


plt.figure(figsize=(14, 7), dpi=300)
plt.plot(np.log(V_agg2_df),
         label=f'Log of Aggregate Portfolio Buy-n-Hold and V(α={alp})',
         color='darkmagenta')
plt.plot(np.log(V2_df),
         label=f'Log of V(α={alp}) Portfolio', color='teal')
plt.plot(log_BH, label='Log of Buy and Hold Portfolio', color='green')
plt.plot(log_rebalance, label='Log of Rebalancing Portfolio',
         color='mediumblue', linestyle='--')
plt.plot(log_market, label='Log of Market Portfolio',
         linestyle='--', color='red')
plt.title(
    'Aggregate Portfolio (B-n-H & α-CK) vs Rebalancing and Market Portfolio')
plt.xlabel('Date')
plt.ylabel('Log of Vt')
plt.legend()
plt.savefig('7_agg1.png')
plt.show()


# ###############################################################
# Special Case
# Logarithmic difference (ln V* - ln V_T)
# Verify as T tends to infinity
k = 2
# V_T represents the aggregated portfolio (BnH)
V_T = BnH

bayes = []
bound = []
for t in range(1, len(returns)):
    # V_star is the best-performing portfolio
    V_star = np.maximum(BnH[t], V2[t])
    log_diff = (np.log(V_star) - np.log(V_T[t])) / t
    upper_bound = np.log(k) / t
    bayes.append(log_diff)
    bound.append(upper_bound)


bayes_df = pd.DataFrame(bayes, index=returns.index[1:])
bound_df = pd.DataFrame(bound, index=returns.index[1:])

plt.figure(figsize=(14, 7), dpi=300)
plt.plot(bayes_df, label='Log(V*) - Log(V) / T', color='royalblue')
plt.plot(bound_df, linestyle='--', color='red',
         label=r'$\ln(k) / T$ upper bound')
plt.ylim(-0.0025, 0.01)
plt.title('Logarithmic Difference between Best and Aggregated Portfolio')
plt.xlabel('Time')
plt.ylabel('Log Difference')
plt.legend()
plt.savefig('bayes.png')
plt.show()


# ############ Task 3.3.2: Learning alpha part 2 ################
# Cover's Portfolio

# Function that produce V(alpha) and theta(alpha) for CK port ###
def compute_v_theta(alpha):
    V_alpha = np.ones(Z)
    theta_alpha = []
    for t in range(1, Z):
        allocation = ck_allocation(returns.iloc[:t], alpha)
        portfolio_return = (allocation * returns.iloc[t]).sum()
        V_alpha[t] = V_alpha[t-1] * (1 + portfolio_return)
        theta_alpha.append(allocation)
    theta_alpha = np.array(theta_alpha)
    return V_alpha, theta_alpha
# ###############################################################


# Compute V(alpha) and theta(alpha) for multiple alphas
alphas3 = np.linspace(0, alp, 20)
V_alphas = []
theta_alphas = []

for alpha in alphas3:
    V_alpha, theta_alpha = compute_v_theta(alpha)
    V_alphas.append(V_alpha)
    theta_alphas.append(theta_alpha)

# Convert lists to numpy arrays for easier manipulation
V_alphas = np.array(V_alphas)
theta_alphas = np.array(theta_alphas)

# Compute Cover's portfolio
V_cover = np.mean(V_alphas, axis=0)
theta_cover = np.mean(theta_alphas, axis=0)
V_cover_df = pd.DataFrame(V_cover, index=returns.index)


plt.figure(figsize=(14, 7), dpi=300)
plt.plot(np.log(V_cover_df), label='Log of Cover\'s Portfolio', color='olive')
plt.plot(log_rebalance, label='Log of Rebalancing Portfolio',
         color='mediumblue', linestyle='--')
plt.plot(log_BH, label='Log of Buy and Hold Portfolio',
         color='green', linestyle='--')
plt.plot(log_market, label='Log of Market Portfolio',
         linestyle='--', color='red')
plt.title('Cover\'s Portfolio vs Rebalancing, BnH and Market Portfolio')
plt.xlabel("Date")
plt.ylabel("Log of Vt")
plt.legend()
plt.savefig('9_cover.png')
plt.show()


# ################## Task 3.3.3 ########################
# Option 1: Cover's Portfolio with m-dimensional simplex
m = returns.shape[1]
num_portfolios = 1000  # Number of portfolios to approximate the simplex

# Generate random weights in the simplex
simplex_weights = np.random.dirichlet(np.ones(m), size=num_portfolios)


# Calculate portfolio values for each simplex weight
def constant_rebalanced_portfolio(weights, returns):
    return (1 + (returns @ weights)).cumprod()


# Calculate portfolio values for each simplex weight
V_b = np.array([constant_rebalanced_portfolio(w, returns)
               for w in simplex_weights])


# Compute theta for Cover's portfolio simplex

theta_cover_simplex = []
for t in range(1, len(returns)):
    num = np.sum(V_b[:, t-1][:, np.newaxis] * simplex_weights, axis=0)
    den = np.sum(V_b[:, t-1])
    theta_cover_simplex.append(num / den)

theta_cover_simplex = np.array(theta_cover_simplex)

# Compute Cover's portfolio
V_cover_simplex = V_b.mean(axis=0)
V_cover_simplex_df = pd.DataFrame(V_cover_simplex, index=returns.index)


plt.figure(figsize=(14, 7), dpi=300)
plt.plot(np.log(V_cover_simplex_df),
         label='Log of Cover\'s Portfolio (Simplex)', color='darkorange')
plt.plot(log_rebalance, label='Log of Rebalancing Portfolio',
         color='mediumblue', linestyle='--')
plt.plot(log_BH, label='Log of Buy and Hold Portfolio',
         color='green', linestyle='--')
plt.plot(log_market, label='Log of Market Portfolio',
         linestyle='--', color='red')
plt.title(
    'Cover\'s Portfolio (Simplex) vs Rebalancing, BnH and Market Portfolio')
plt.xlabel('Date')
plt.ylabel('Log of Vt')
plt.legend()
plt.savefig('10_simplex.png')
plt.show()


# ############### Final Comparison ###############
plt.figure(figsize=(14, 7), dpi=300)
# plt.plot(np.log(V_agg_df),
#        label=f'Log of Aggregate Portfolio V1 = 1 and V2(α={alp})', color='purple')
plt.plot(np.log(V_agg2_df),
         label=f'Log of Aggregate Portfolio Buy-n-Hold and V(α={alp})',
         color='darkmagenta')
plt.plot(np.log(V_cover_df), label='Log of Cover\'s Portfolio', color='olive')
plt.plot(np.log(V_cover_simplex_df),
         label='Log of Cover\'s Portfolio (Simplex)', color='darkorange')
plt.plot(log_rebalance, label='Log of Rebalancing Portfolio', color='mediumblue')
plt.plot(log_BH, label='Log of Buy and Hold Portfolio', color='green')
plt.plot(log_market, label='Log of Market Portfolio',
         linestyle='--', color='red')
plt.title(
    'Comparison between all Strategies')
plt.xlabel('Date')
plt.ylabel('Log of Vt')
plt.legend()
plt.savefig('11_final.png')
plt.show()
# ###############################################


# ############################################################
# Theta-plots for Aggregate, Cover and Cover Simplex portfolio
# ############################################################
fig, axs = plt.subplots(2, 1, figsize=(14, 14), dpi=300)

# 1. Aggregate Portfolio
axs[0].set_title(f'Theta Plot for Aggregate Portfolio (B-n-H & V(α={alp})')
for i in range(m):
    axs[0].plot(returns.index[1:], theta_agg[:, i], label=stocks[i])
axs[0].plot(returns.index[1:], 1 - np.sum(theta_agg, axis=1),
            label='Cash', linestyle='--')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Weight')
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs[0].set_ylim(0, 1)

# 2. Cover's Portfolio
axs[1].set_title('Theta Plot for Cover\'s Portfolio')
for i in range(m):
    axs[1].plot(returns.index[1:], theta_cover[:, i], label=stocks[i])
axs[1].plot(returns.index[1:], 1 - np.sum(theta_cover, axis=1),
            label='Cash', linestyle='--')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Weight')
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('12_theta2.png')
plt.show()


# ##############################################
# Maximum Drawdown for all strategies
# ##############################################
mdd_uniform = MDD(rebalance_port)
mdd_buyandh = MDD(BnH)
mdd_market = MDD(market_port)
print(f"Maximum Drawdown for Rebalancing Portfolio: {mdd_uniform}")
print(f"Maximum Drawdown for Buy and Hold Portfolio: {mdd_buyandh}")
print(f"Maximum Drawdown for Market Portfolio: {mdd_market}")

mdd_Wt_reb = MDD(Wt_rebalance)
mdd_Wt_bnh = MDD(Wt_bnh)
mdd_Wt_mkt = MDD(Wt_market)
print(f"Maximum Drawdown for PoP Rebalancing Portfolio: {mdd_Wt_reb}")
print(f"Maximum Drawdown for PoP Buy and Hold Portfolio: {mdd_Wt_bnh}")
print(f"Maximum Drawdown for PoP Market Portfolio: {mdd_Wt_mkt}")

for alpha in alphas:
    mdd1 = MDD(V_df[alpha])
    print(f'Maximum Drawdown for α-CK Allocation Portolio (α={alpha}): {mdd1}')

for alpha in alphas:
    avdd1 = MDD(Wt_df[alpha])
    print(
        f'Maximum Drawdown for α-CK PoP (α={alpha}): {avdd1}')

mdd_V_agg = MDD(V_agg2_df)
print(
    f"Maximum Drawdown for Aggregate Portfolio BnH and V(α={alp}): {mdd_V_agg[0]}")

mdd_cover = MDD(V_cover_df)
print(f'Maximum Drawdown for Cover\'s Portfolio: {mdd_cover[0]}')

mdd_cover_simplex = MDD(V_cover_simplex_df)
print(
    f'Maximum Drawdown for Cover\'s Portfolio (Simplex): {mdd_cover_simplex[0]}')


# ##############################################
# Average Drawdown for all strategies
# ##############################################
avdd_uniform = AVDD(rebalance_port)
avdd_buyandh = AVDD(BnH)
avdd_market = AVDD(market_port)
print(f"Average Drawdown for Rebalancing Portfolio: {avdd_uniform}")
print(f"Average Drawdown for Buy and Hold Portfolio: {avdd_buyandh}")
print(f"Average Drawdown for Market Portfolio: {avdd_market}")

avdd_Wt_reb = AVDD(Wt_rebalance)
avdd_Wt_bnh = AVDD(Wt_bnh)
avdd_Wt_mkt = AVDD(Wt_market)
print(f"Average Drawdown for PoP Rebalancing Portfolio: {avdd_Wt_reb}")
print(f"Average Drawdown for PoP Buy and Hold Portfolio: {avdd_Wt_bnh}")
print(f"Average Drawdown for PoP Market Portfolio: {avdd_Wt_mkt}")

for alpha in alphas:
    avdd1 = AVDD(V_df[alpha])
    print(
        f'Average Drawdown for α-CK Allocation Portolio (α={alpha}): {avdd1}')

for alpha in alphas:
    avdd1 = AVDD(Wt_df[alpha])
    print(
        f'Average Drawdown for α-CK PoP (α={alpha}): {avdd1}')

avdd_V_agg = AVDD(V_agg2_df)
print(
    f"Average Drawdown for Aggregate Portfolio BnH and V(α={alp}): {avdd_V_agg[0]}")

avdd_cover = AVDD(V_cover_df)
print(f'Average Drawdown for Cover\'s Portfolio: {avdd_cover[0]}')

avdd_cover_simplex = AVDD(V_cover_simplex_df)
print(
    f'Average Drawdown for Cover\'s Portfolio (Simplex): {avdd_cover_simplex[0]}')
