function Backtest 

    % input data
    assetData = "AssetPrices.csv";
    factorData = "FactorReturns.csv";
    fxData = "ExchangeRate.csv";

    % input parameters
    initialVal = 100000;
    US = [ones(1,7) zeros(1,7) ones(1,1)];
    alpha = 0.95;
    
    % load the assets prices
    adjClose = readtable(assetData);
    adjClose.Properties.RowNames = cellstr(datetime(adjClose.Date));
    adjClose.Properties.RowNames = cellstr(datetime(adjClose.Properties.RowNames));
    adjClose.Date = [];
    
    % load the factors return 
    factorRet = readtable(factorData);
    factorRet.Properties.RowNames = cellstr(datetime(factorRet.Date));
    factorRet.Properties.RowNames = cellstr(datetime(factorRet.Properties.RowNames));
    factorRet.Date = [];
    
    % load the exchange rate CAD/USD
    fxRate = readtable(fxData);
    fxRate.Properties.RowNames = cellstr(datetime(fxRate.Date));
    fxRate.Properties.RowNames = cellstr(datetime(fxRate.Properties.RowNames));
    fxRate.Date = [];
    
    riskFree = factorRet(:,8); 
    factorRet = factorRet(:,1:7);
    
    % identify the tickers and the dates 
    tickers = adjClose.Properties.VariableNames';
    dates   = datetime(factorRet.Properties.RowNames);
    
    % calculate the asset's exceed returns 
    prices  = table2array(adjClose);
    returns = (prices(2:end,:) - prices(1:end-1,:)) ./ prices(1:end-1,:);
    returns = returns - (diag(table2array(riskFree)) * ones(size(returns)));
    returns = array2table(returns);
    returns.Properties.VariableNames = tickers;
    returns.Properties.RowNames = factorRet.Properties.RowNames;
    
    % align the price and exchange rate table to the factor returns tables by
    % discarding the first observation.
    adjClose = adjClose(2:end,:);
    fxRate = fxRate(2:end,:);
    
    % start and end of the investment period 
    testStart = datetime(returns.Properties.RowNames{1});
    testEnd = testStart + calyears(1) - days(1);
    
    % parameter calibration period
    calStart = datetime(returns.Properties.RowNames{1}) + calyears(1);
    calEnd = calStart + calyears(4) - days(1);
    
    periodReturns = table2array(returns(calStart <= dates & dates <= calEnd,:));
    periodFactRet = table2array(factorRet(calStart <= dates & dates <= calEnd,:));
    
    [mu,Q] = FactorModel(periodReturns, periodFactRet);
    x = PortfolioWeight(mu, Q, US);
    
    NoShares = x .* initialVal ./ ...
            (prices(1,:) .* (1 - US + US * table2array(fxRate(1,1))))'; 
    
    % Out-of-sample performance    
    testPrice = table2array(adjClose(testStart <= dates & dates <= testEnd,:));
    testFX = table2array(fxRate(testStart <= dates & dates <= testEnd,:));
    testVal = testPrice .* (1 - US + US .* testFX) * NoShares;
    figure();
    plotDates = dates(dates < datetime(returns.Properties.RowNames{1}) + calyears(1));
    plot(plotDates, testVal);
    title('Out-of-Sample Performance');
    
    VaR = testVal .* (mu'*x + sqrt(x'*Q*x)*norminv(alpha));
    CVaR = testVal .* (mu'*x + 1/(1-alpha) * sqrt(x'*Q*x) * ...
        normpdf(norminv(alpha)));
    Loss = - testVal(2:end) + testVal(1:end-1);
    
    figure();
    plot(plotDates(2:end), VaR(2:end));
    hold on
    plot(plotDates(2:end), CVaR(2:end));
    plot(plotDates(2:end), Loss);
    hold off
    legend('VaR', 'CVaR', 'Loss', 'Location', 'northwest')
    title('Backtesting VaR and CVaR')
    
    % In-sample performance
    testPrice = table2array(adjClose(calStart <= dates & dates <= calEnd,:));
    testFX = table2array(fxRate(calStart <= dates & dates <= calEnd,:));
    testVal = testPrice .* (1 - US + US .* testFX) * NoShares;
    figure();
    plotDates = dates(calStart <= dates & dates <= calEnd);
    plot(plotDates, testVal);
    title('In-Sample Performance')
      
end

% The function FactorModel is used to determine the inputs of the
% portfolio weight optimization problem 
% returns: asset returns 
% factRet: risk factor returns 
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


% The function PortfolioWeight is used to determine the assets weight (MVO)
% mu: assets expected returns
% Q: variance-covariance matrix of the asset returns 
% US: indicator of assets in USD
function x = PortfolioWeight(mu, Q, US)

    % Find the total number of assets
    n = size(Q,1); 
    
    % Set the target as the average expected return of all assets
    targetRet = mean(mu);
    
    % Disallow short sales
    lb = ones(n,1)*0.01;
    ub = ones(n,1);

    % Add the expected return constraint
    A = -1 .* mu';
    b = -1 * targetRet;

    % Constrain weights to sum to 1
    Aeq = [ones(1,n); US];
    beq = [1; 0.5];

    % Set the quadprog options 
    options = optimoptions( 'quadprog', 'TolFun', 1e-9, 'Display','off');
    
    % Optimal asset weights
    x = quadprog( 2 * Q, [], A, b, Aeq, beq, lb, ub, [], options);

end