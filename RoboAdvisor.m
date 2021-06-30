function RoboAdvisor
   
    % input data
    assetData = "AssetPrices.csv";
    factorData = "FactorReturn.csv";
    fxRate = "ExchangeRate.csv";

    % input parameters
    initialVal = 100000;
    injection = 10000;
    investPeriod = 6;
    transactionFee = ;
    
    
    

end

% The function factor model is used to determine the inputs of the
% portfolio weight optimization problem  
function [mu, Q] = FactorModel(returns, factRet)

    % Number of observations and factors
    [T, p] = size(factRet); 
    
    % Data matrix
    X = [ones(T,1) factRet];
    
    % Regression coefficients
    B = (X' * X) \ X' * returns;
    
    % Separate B into alpha and betas
    a = B(1,:)';     
    V = B(2:end,:); 
    
    % Residual variance
    ep       = returns - X * B;
    sigma_ep = 1/(T - p - 1) .* sum(ep .^2, 1);
    D        = diag(sigma_ep);
    
    % Calculate the asset expected returns and covariance matrix
    f_bar = mean(factRet,1)';
    F     = cov(factRet);
    mu = a + V' * f_bar;
    Q  = V' * F * V + D;
    
    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    Q = (Q + Q')/2;
    
end

% The function PortfolioWeight is used to determine the assets weight
function PortfolioWeight

end