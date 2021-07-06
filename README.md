# MMF2025

There are three main sections of the current version. 

- RoboAdvisor: This function includes the main processes of the automatic rebalancing and the performance tracking. Rebalancing would be done semi-annually after each injection. Also, breaching the 40/60 currency limit would trigger the rebalance process. The function will produce the semi-annual Sharpe ratios, asset allocation plots and portfolio value plots.

- FactorModel: This function is used to calculate the input of the optimization problem. We will use OLS to calculated the expected returns of the asset with multiple risk factors. In addition, this function also produces variance-covariance matrix. This matrix includes the information on the correlations between returns and the correlation between the risk factors. The inputs of this function are the historical prices of the assets and the historical returns of the risk factors.

- PortfolioWeight: This function is used to calculate the asset weights with the MVO maximization model. We add a limit on the asset weights to make sure that the ratio of the exposure to CAD and USD is 50/50 on each rebalancing day. 
