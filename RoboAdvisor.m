function RoboAdvisor
   
    % input data
    assetData = "AssetPrices.csv";
    factorData = "FactorReturns.csv";
    fxData = "ExchangeRate.csv";

    % input parameters
    alpha = 0.95;
    
    initialVal = 100000;
    injection = 10000;
    investPeriod = 6;
    transactionFee = 2000;
    US = [ones(1,7) zeros(1,7) ones(1,1)];
    
    Equity = [ones(1,9) zeros(1,3), ones(1,1) zeros(1,2)];
    Bond = [zeros(1,9) ones(1,2) zeros(1,4)];
    Cash = [zeros(1,13) ones(1,2)];
    Commodity = [zeros(1,11) ones(1,1) zeros(1,3)];
    
    
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
    
    % load scenario 1 data 
    scenario1 = "ScenarioAnalysis_Extreme.csv";
    factorRet1 = readtable(scenario1);
    factorRet1.Properties.RowNames = cellstr(datetime(factorRet1.Date));
    factorRet1.Properties.RowNames = cellstr(datetime(factorRet1.Properties.RowNames));
    factorRet1.Date = [];
    factorRet_S1 = table2array(factorRet1(:,1:7));
   
    % load scenario 2 data 
    scenario2 = "ScenarioAnalysis_UP.csv";
    factorRet2 = readtable(scenario2);
    factorRet2.Properties.RowNames = cellstr(datetime(factorRet2.Date));
    factorRet2.Properties.RowNames = cellstr(datetime(factorRet2.Properties.RowNames));
    factorRet2.Date = [];
    factorRet_S2 = table2array(factorRet2(:,1:7));
    
    % load scenario 3 data 
    scenario3 = "ScenarioAnalysis_DOWN.csv";
    factorRet3 = readtable(scenario3);
    factorRet3.Properties.RowNames = cellstr(datetime(factorRet3.Date));
    factorRet3.Properties.RowNames = cellstr(datetime(factorRet3.Properties.RowNames));
    factorRet3.Date = [];
    factorRet_S3 = table2array(factorRet3(:,1:7));
    
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
    
    % preallocate space for VaR and CVaR of the portfolio 
    VaR = zeros(1, investDays);
    CVaR = zeros(1, investDays);
    SVaR1 = zeros(1, investDays);
    SVaR2 = zeros(1, investDays);
    SVaR3 = zeros(1, investDays);
    
    % record the rebalance date position 
    rb_date = zeros(1, investDays);
    
    % rebalancing process
    for t = 1:NoPeriods+1
        
        periodReturns = table2array(returns(calStart <= dates & dates <= calEnd,:));
        periodFactRet = table2array(factorRet(calStart <= dates & dates <= calEnd,:));
        
        currentPrices = table2array(adjClose(countDay,:))';
        currentRate = table2array(fxRate(countDay,1));
        
        [mu,Q] = FactorModel(periodReturns, periodFactRet);
        [Smu1, SQ1] = ScenarioAnalysis(periodReturns, periodFactRet, factorRet_S1);
        [Smu2, SQ2] = ScenarioAnalysis(periodReturns, periodFactRet, factorRet_S2);
        [Smu3, SQ3] = ScenarioAnalysis(periodReturns, periodFactRet, factorRet_S3);
        
        rb_date(countDay) = 1;
        
        if countDay == 1
            currentVal(countDay) = initialVal;
            x(:,countDay) = PortfolioWeight(mu, Q, ones(n,1)*1/n, US);
        else    
            currentVal(countDay) = currentPrices' .* (1 - US + US * currentRate) ...
                * NoShares + injection - transactionFee;
            x(:,countDay) = PortfolioWeight(mu, Q, x(:, countDay-1), US);
        end
        
        NoShares = x(:,countDay) .* currentVal(countDay) ./ ...
            (currentPrices .* (1 - US + US * currentRate)'); 
        
        NoDays = sum(testStart <= dates & dates <= testEnd);
        periodVal = zeros(NoDays,1);
        periodVal(1) = currentVal(countDay);
        
        USDAllocation = US * x(:,countDay);
        allocation(countDay,1) = USDAllocation; 
        allocation(countDay,2) = 1 - USDAllocation;
        
        % calculate VaR and CVaR on daily basis
        VaR(countDay) = currentVal(countDay) * (mu'*x(:,countDay) +...
            sqrt(x(:,countDay)'*Q*x(:,countDay))*norminv(alpha));
        CVaR(countDay) = currentVal(countDay) * (mu'*x(:,countDay) + 1/(1-alpha) * ...
            sqrt(x(:,countDay)'*Q*x(:,countDay)) * normpdf(norminv(alpha)));
        
        % calculate VaR under stressed scenario
        SVaR1(countDay) = currentVal(countDay) * (Smu1'*x(:,countDay) +...
            sqrt(x(:,countDay)'*SQ1*x(:,countDay))*norminv(alpha));
        SVaR2(countDay) = currentVal(countDay) * (Smu2'*x(:,countDay) +...
            sqrt(x(:,countDay)'*SQ2*x(:,countDay))*norminv(alpha));
        SVaR3(countDay) = currentVal(countDay) * (Smu3'*x(:,countDay) +...
            sqrt(x(:,countDay)'*SQ3*x(:,countDay))*norminv(alpha));

        for i = 2:NoDays
            countDay = countDay + 1;
            currentPrices = table2array(adjClose(countDay,:))';
            currentRate = table2array(fxRate(countDay,1)); 
            
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
                
                [mu,Q] = FactorModel(periodReturns, periodFactRet);
                [Smu1, SQ1] = ScenarioAnalysis(periodReturns, ...
                    periodFactRet, factorRet_S1);
                [Smu2, SQ2] = ScenarioAnalysis(periodReturns, ...
                    periodFactRet, factorRet_S2);
                [Smu3, SQ3] = ScenarioAnalysis(periodReturns, ...
                    periodFactRet, factorRet_S3);
                
                x(:,countDay) = PortfolioWeight(mu, Q, x(:, countDay-1), US);
                NoShares = x(:,countDay) .* currentVal(countDay) ./ ...
                    (currentPrices .* (1 - US + US * currentRate)'); 
                
                USDAllocation = US * x(:,countDay);
                allocation(countDay,1) = USDAllocation; 
                allocation(countDay,2) = 1 - USDAllocation;
            else 
                x(:,countDay) = currentPrices .* (1 - US + US * currentRate)' ...
                    .* NoShares ./ currentVal(countDay);
                allocation(countDay,1) = USDAllocation; 
                allocation(countDay,2) = 1 - USDAllocation;
            end
           
            periodVal(i) = currentVal(countDay);
            
            % Calculated VaR and CVaR on daily based
            VaR(countDay) = currentVal(countDay) * (mu'*x(:,countDay) +...
                sqrt(x(:,countDay)'*Q*x(:,countDay))*norminv(alpha));
            CVaR(countDay) = currentVal(countDay) * (mu'*x(:,countDay) + 1/(1-alpha) * ...
                sqrt(x(:,countDay)'*Q*x(:,countDay)) * normpdf(norminv(alpha)));
            
            % calculate VaR under stressed scenario
            SVaR1(countDay) = currentVal(countDay) * (Smu1'*x(:,countDay) +...
                sqrt(x(:,countDay)'*SQ1*x(:,countDay))*norminv(alpha));
            SVaR2(countDay) = currentVal(countDay) * (Smu2'*x(:,countDay) +...
                sqrt(x(:,countDay)'*SQ2*x(:,countDay))*norminv(alpha));
            SVaR3(countDay) = currentVal(countDay) * (Smu3'*x(:,countDay) +...
                sqrt(x(:,countDay)'*SQ3*x(:,countDay))*norminv(alpha));
            
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
        rf = table2array(riskFree(testStart <= dates & dates <= testEnd,1));
        portExRets = portRets - rf(2:end);
        SR = (geomean(portExRets + 1) - 1) / std(portExRets);
        disp(['Sharpe ratio (', num2str(t), ') :', num2str(SR)]);
        
        % calculate annulized return
        AR = (periodVal(end)/periodVal(1))^2-1;
        disp(['Annulized return (', num2str(t), ') :', num2str(AR)]);
        
        % plot the asset class and geographic allcation before each injection 
        figure();
        pie([Equity; Bond; Cash; Commodity] * x(:,countDay-(NoDays-1)), ...
            {'Equity', 'Bond', 'Cash', 'Commodity'})
        title(['Asset Allocation (', num2str(t), ')'])
        
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
    title('Currencies Allcation');
    
    % plot the portfolio value over the entire period 
    figure();
    plot(plotDates, currentVal);
    datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
    set(gca,'XTickLabelRotation',30);
    title('Portfolio Value');
    
    % plot one-day 95% VaR and ES
    figure();
    plot(plotDates, VaR);
    hold on
    plot(plotDates, CVaR);
    hold off
    legend('VaR', 'CVaR');
    title('95% VaR & ES')
    
    % plot the PnL
    PnL = currentVal(2:end) - currentVal(1:end-1);
    rb_date = rb_date(2:end)';
    PnL = PnL - injection * rb_date;
    figure();
    plot(plotDates(2:end), PnL);
    title('Daily PnL')
    
    % plot the cumulative return
    Return = PnL ./ currentVal(1:end-1);
    cum_return = cumprod(1+Return);
    plot(plotDates(2:end), cum_return);
    
    AOA = "AOA.csv";
    benchmark = readtable(AOA);
    benchmark.Properties.RowNames = cellstr(datetime(benchmark.Date));
    benchmark.Properties.RowNames = cellstr(datetime(benchmark.Properties.RowNames));
    benchmark.Date = [];
    benchmark = table2array(benchmark);
    benchmark_return = benchmark(2:end,2);
    benchmark_cum = cumprod(1+benchmark_return);
    
    hold on
    plot(plotDates(2:end), benchmark_cum)
    hold off
    legend('Robo Advisor', 'Benchmark (AOA)', 'Location', 'northwest')
    title('Cumulative Return')
    
    % plot VaR under stressed scenarios 
    figure();
    plot(plotDates, SVaR1);
    hold on
    plot(plotDates, SVaR2);
    plot(plotDates, SVaR3);
    hold off
    legend('VaR (Scenario 1)', 'VaR (Scenario 2)','VaR (Scenario 3)');
    title('VaR under Different Scenarios');
    
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
function x = PortfolioWeight(mu, Q, x0, US)

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


% The function PortfolioWeight2 is used to determine the assets weight (Robust MVO)
% mu: assets expected returns
% Q: variance-covariance matrix of the asset returns 
% US: indicator of assets in USD
function  x = PortfolioWeight2(mu, Q, x0, US)
    
    lambda = 1;
    
    % Define the parameters for the robust MVO
    alpha = 0.95;
    
    % Find the number of assets
    n = size(Q,1);
    
    % Define the radius of our uncertainty set
    ep = sqrt( chi2inv(alpha, n) );
    
    % Find the value of Theta (i.e., the squared standard error of our
    % expected returns)
    Theta = diag( diag(Q) ) ./ (252*5);

    % Square root of Theta
    sqrtTh = sqrt(Theta);
    
    % Linear equality Constraint bounds
    Aeq = [ones(1,n); US];
    beq = [1; 0.5]; 
    
    % Linear inequality Constraint bounds
    b = [];
    A = []; 
    
    % Lower and upper bounds on variables
    lb = ones(n,1)*0.01;
    ub = 3/n*ones(n,1);
    
    % It might be useful to increase the tolerance of 'fmincon'
    options = optimoptions('fmincon', 'TolFun', 1e-9, 'Display','off');

    % Solve using fmincon
    x = fmincon(@(x)objFun(x, mu, Q, lambda, sqrtTh, ep),x0,A,b,Aeq,beq,lb,ub,...
            @(x)nonlcon(x), options);
    
end
% Defind the objective function
function f = objFun(x, mu, Q, lambda, sqrtTh, ep)
    f = (lambda * x' * Q * x) - mu' * x + ep * norm( sqrtTh * x ); 
end
    % Defind the equality and inequality nonlinear constraints
function [c, ceq] = nonlcon(x)
    c = [];
    ceq = [];
end

% The function ScenarioAnalysis is used to estimate the return and 
% covariance matrix under different scenarios
function [Smu, SQ] = ScenarioAnalysis(returns, factRet, SfactRet)

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
    f_bar = mean(SfactRet,1)';
    F     = cov(SfactRet);
    Smu = a + V' * f_bar;
    SQ  = V' * F * V + D;
    
    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    SQ = (SQ + SQ')/2;

end
