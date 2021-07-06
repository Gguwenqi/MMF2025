function RoboAdvisor
   
    % input data
    assetData = "AssetPrices.csv";
    factorData = "FactorReturns.csv";
    fxData = "ExchangeRate.csv";

    % input parameters
    initialVal = 100000;
    injection = 10000;
    investPeriod = 6;
    transactionFee = 2000;
    US = ;
    
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
    
    riskFree = factorRet(:,); 
    factorRet = factorRet(:,1:);
    
    % identify the tickers and the dates 
    tickers = adjClose.Properties.VariableNames';
    dates   = datetime(factorRet.Properties.RowNames);
    
    % calculate the asset's exceed returns 
    prices  = table2array(adjClose);
    returns = (prices(2:end,:) - prices(1:end-1,:)) ./ prices(1:end-1,:);
    returns = returns - (diag(table2array(riskFree)) * ones(size(returns)));
    returns = array2table(returns);
    returns.Properties.VariableNames = tickers;
    returns.Properties.RowNames = cellstr(datetime(adjClose.Properties.RowNames));
    
    % align the price and exchange rate table to the factor returns tables by
    % discarding the first observation.
    adjClose = adjClose(2:end,:);
    fxRate = fxRate(2:end,:);
    
    % total number of half year period 
    NoPeriods = 10;
    
    % number of assets
    n = size(adjClose,2);
    
    % start and end of the investment period (before injection)
    testStart = datetime(returns.Properties.RowNames{1}) + calyears(5);
    testEnd = testStart + calmonths(investPeriod) - days(1);
    
    % parameter calibration period
    calStart = datetime(returns.Properties.RowNames{1});
    calEnd = testStart - days(1);
    
    % preallocate space for the currency allocation
    investDays = sum(testStart <= dates);
    allocation = zeros(investDays, 2);
    countDay = 1;
    
    % preallocate space for the portfolio per period value (in CAD)
    currentVal = zeros(investDays, 1);
    
    % preallocate space for the portfolio asset weight
    x = zeros(n, investDays);
    
    % rebalancing process
    for t = 1:NoPeriods+1
        
        periodReturns = table2array(returns(calStart <= dates & dates <= calEnd,:));
        periodFactRet = table2array(factorRet(calStart <= dates & dates <= calEnd,:));
        
        currentPrices = table2array(adjClose(countDay,:))';
        currentRate = table2array(fxRate(countDay));
        
        if countDay == 1
            currentVal(countDay) = initialVal;
        else    
            currentVal(countDay) = currentPrices' .* (1 - US + US * currentRate) ...
                * NoShares + injection - transactionFee;
        end
        
        x(:,countDay) = PortfolioWeight(FactorModel(periodReturns, periodFactRet));
        NoShares = x(:,countDay) .* currentVal(countDay) ./ ...
            (currentPrices .* (1 - US + US * currentRate)'); 
        
        NoDays = sum(testStart <= dates & dates <= testEnd);
        periodVal = zeros(NoDays,1);
        periodVal(1) = currentVal(countDay);

        for i = 2:NoDays
            countDay = countDay + 1;
            currentPrices = table2array(adjClose(countDay,:))';
            currentRate = table2array(fxRate(countDay)); 
            
            currentVal(countDay) = currentPrices' .* ...
                (1 - US + US * currentRate) * NoShares; 
            
            USDVal = currentPrices' .* (US * currentRate) * NoShares;
            USDAllocation = USDVal/currentVal(countDay);
            
            % trigger rebalance if the currency allocation break the limit 
            if USDAllocation<0.4 || USDAllocation>0.6
                calEnd = datetime(returns.Properties.RowNames{...
                    find(dates == testStart,1) + countDay - 1}); 
                calStart = calEnd - calyears(5);
               
                periodReturns = table2array(returns(calStart <= dates & dates <= calEnd,:));
                periodFactRet = table2array(factorRet(calStart <= dates & dates <= calEnd,:));
                
                currentVal(countDay) = currentPrices' .* (1 - US + US * currentRate) ...
                    * NoShares - transactionFee; 
                
                x(:,countDay) = PortfolioWeight(FactorModel(periodReturns, periodFactRet));
                NoShares = x(:,countDay) .* currentVal(countDay) ./ ...
                    (currentPrices .* (1 - US + US * currentRate)'); 
                
                USDAllocation = US * x(:,countDay);
                allocation(countDay,1) = USDAllocation; 
                allocation(countDay,2) = 1 - USDAllocation;
            else 
                x(:,countDay) = x(:,countDay-1);
                allocation(countDay,1) = USDAllocation; 
                allocation(countDay,2) = 1 - USDAllocation;
            end
           
            periodVal(i) = currentVal(countDay);    
        end
        
        % Plot semi-annully value graph
        plotDates = dates(testStart <= dates & dates <= testEnd);
        figure();
        plot(plotDates, periodVal);
        datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
        set(gca,'XTickLabelRotation',30);
        title('Portfolio Wealth Evolution');
        ylabel('Total wealth (CAD)')

        % calculate semi-annully Sharpe Ratio
        portRets = periodVal(2:end) ./ periodVal(1:end-1) - 1;
        rf = table2array(riskFree(testStart <= dates & dates <= testEnd));
        portExRets = portRets - rf(2:end);
        SR = (geomean(portExRets + 1) - 1) / std(portExRets);
        disp(['Sharpe ratio (', num2str(t), ') :', num2str(SR)]);
        
        % plot the asset allcation before each injection 
        figure();
        pie(x(:,countDay));
        legend(tickers);
        
        % update the investment period 
        testStart = testStart + calmonths(investPeriod);
        testEnd = testStart + calmonths(investPeriod) - days(1);
        calEnd = testStart - days(1);
        calStart = calEnd - calyears(5);
        
        countDay = countDay + 1;
        
    end
    
    % plot the currency allocation over the entire period 
    plotDates = dates(dates >= datetime(returns.Properties.RowNames{1}) + calyears(5));
    figure();
    plot(plotDates, allocation(:,1));
    hold on
    plot(plotDates, allocation(:,2));
    hold off
    datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
    set(gca,'XTickLabelRotation',30);
    legend('USD Weight', 'CAD Weight');
    title('Currencies Allcation over the Entire Period');
    
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



% The function PortfolioWeight is used to determine the assets weight
% mu: assets expected returns
% Q: variance-covariance matrix of the asset returns 
% US: indicator of assets in USD
function x = PortfolioWeight(mu, Q, US)

    % Find the total number of assets
    n = size(Q,1); 
    
    % Set the target as the average expected return of all assets
    targetRet = mean(mu);
    
    % Disallow short sales
    lb = zeros(n,1);

    % Add the expected return constraint
    A = -1 .* mu';
    b = -1 * targetRet;

    % Constrain weights to sum to 1
    Aeq = [ones(1,n); US];
    beq = [1; 0.5];

    % Set the quadprog options 
    options = optimoptions( 'quadprog', 'TolFun', 1e-9, 'Display','off');
    
    % Optimal asset weights
    x = quadprog( 2 * Q, [], A, b, Aeq, beq, lb, [], [], options);

end