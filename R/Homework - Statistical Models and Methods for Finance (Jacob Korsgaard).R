## Statistical Models and Methods for Finance
## Homework: January 26, 2026
## By: Jacob Korsgaard
## Examinator: Stefano Tonellato 

## 0 Clear everything and library ############################################
# Clear Everything 
cat("\014")
rm(list = ls())
graphics.off()

# Load Required Packages
library(quantmod) # To get data from Yahoo
require(forecast) # To make forecasts
library(urca) # Unit root test
library(fGarch)
library(timeSeries)
library(TSA)

## 1 Download Multivariate Time-Series #######################################
# Tesla Inc. Stock
getSymbols('TSLA', from = '2024-10-01', to = '2026-01-22')

# Initial Overview of Data
summary(TSLA) # Summary of time-series
class(TSLA) #"xts", "zoo" as expected
str(TSLA) # ts, time series data
head(TSLA) # First days of multivariate time series
tail(TSLA) # Last days of multivariate time series

# Create Train/Test Split of data
# Training Data
TSLA_train_high = TSLA$TSLA.High['2024-10-01/2026-01-12']
TSLA_train_low = TSLA$TSLA.Low['2024-10-01/2026-01-12']
TSLA_train = TSLA_train_high - TSLA_train_low

# Test Data
TSLA_test_high = TSLA$TSLA.High['2026-01-13/2026-01-22']
TSLA_test_low = TSLA$TSLA.Low['2026-01-13/2026-01-22']
TSLA_test = TSLA_test_high - TSLA_test_low

# Check the Results
head(TSLA_train_high); head(TSLA_train_low); head(TSLA_train)
head(TSLA_test_high); head(TSLA_test_low); head(TSLA_test)

# Summary of the Results
summary(TSLA_train); summary(TSLA_test)

## 1 Examination of Time Series ###############################################
## 1.1 Plot Raw Data ##########################################################
# Setup for Plotting Layout
par(mfrow = c(3, 1))

# Plot Raw Data 
plot(TSLA_train, main = "Raw Data: Daily Range TESLA")
acf(TSLA_train, main = "ACF of Daily Range TESLA")
pacf(TSLA_train, main = "PACF of Daily Range TESLA")

# Reset Plotting Layout
par(mfrow = c(1, 1))

## 1.2 Augmented Dickey-Fuller (ADF) Test #####################################
# Tests for Unit Root 
adf_raw = ur.df(TSLA_train, type = 'none', selectlags = 'BIC')
print(summary(adf_raw)) # Conclusion: stationary at all significance levels

## 2 Time-Series Modelling ####################################################
## 2.1 Autoregressive Model of Order 1 ########################################
# Fit an ARIMA(1,0,0) model
AR1 = Arima(ts(as.numeric(TSLA_train)), order = c(1,0,0)) 

# Print Results
print(AR1)
tsdiag(AR1) # Non-satisfactory

## 2.2 Autoregressive Model of Order 2 ########################################
# Fit an ARIMA(2,0,0) model
AR2 = Arima(ts(as.numeric(TSLA_train)), order = c(2,0,0))

# Print Results
print(AR2)
tsdiag(AR2) # Satisfactory

## 2.3 Auto.arima Function using BIC ##########################################
## Fit Model using auto.arima
auto.model.bic = auto.arima(TSLA_train, seasonal = FALSE, 
                            stepwise = FALSE,
                            max.p = 5, # Max AR lags
                            max.q = 5, # Max MA lags
                            ic = 'bic', # BIC tends to select a more parsimonious model
                            test = 'adf') # ADF to guide stationarity

# Print Results
print(auto.model.bic)
tsdiag(auto.model.bic) # Satisfactory

## 2.4 Test of Coefficients ####################################################
## AR(2) Model Statistical Test
# Extract coefficients and standard errors
coef_AR2 = coef(AR2)
se_AR2 = sqrt(diag(vcov(AR2)))

# Compute z-statistics for AR coefficients
z_phi1 = coef_AR2["ar1"] / se_AR2["ar1"]
z_phi2 = coef_AR2["ar2"] / se_AR2["ar2"]

# Display results
z_AR2 = c(z_phi1, z_phi2)
names(z_AR2) = c("z_phi1", "z_phi2")
print(z_AR2)

## ARMA(1,1) Model Statistical Test
# Extract coefficients and standard errors
coef_ARMA = coef(auto.model.bic)
se_ARMA = sqrt(diag(vcov(auto.model.bic)))

# Compute z-statistics for ARMA coefficients
z_phi = coef_ARMA["ar1"] / se_ARMA["ar1"]
z_theta = coef_ARMA["ma1"] / se_ARMA["ma1"]

# Display results
z_ARMA = c(z_phi, z_theta)
names(z_ARMA) = c("z_phi", "z_theta")
print(z_ARMA)

## 3 Forecasting ##############################################################
# Horizons for train/test split
nrow(TSLA_train)
nrow(TSLA_test)

## 3.1 Forecast from AR(2) Model #############################################
# Forecast using AR(2) model
forecasts.AR2 = forecast(AR2)
print(forecasts.AR2) # Initial look

# Generate a 6-step-ahead forecast with a 95% confidence level
predfit.AR2 = forecast(AR2, h = nrow(TSLA_test), level = 95)

# Convert test data into time series object
TSLA.test.AR2 = ts(as.numeric(TSLA_test), start = time(predfit.AR2$mean)[1], frequency = 1)

# Plot the forecast, including the last 6 observed points
plot(predfit.AR2, include = 6, showgap = FALSE, main = "Forecasts from the Fitted AR(2) Model")
points(TSLA.test.AR2, col = "firebrick4", pch = 19)

# Metrics for forecast accuracy (test set vs predicted mean)
accuracy(predfit.AR2$mean, TSLA_test)

## 3.2 Forecast from auto.arima, based on BIC #################################
# Forecast using ARMA(1,1) model
forecasts.bic = forecast(auto.model.bic)
print(forecasts.bic) # Initial look

# Generate a 6-step-ahead forecast with a 95% confidence level
predfit.bic = forecast(auto.model.bic, h = nrow(TSLA_test), level = 95)

# Convert test data into time series object
TSLA.test.bic = ts(as.numeric(TSLA_test), start = time(predfit.bic$mean)[1], frequency = 1)

# Plot the forecast, including the last 6 observed points
plot(predfit.bic, include = 6, showgap = FALSE, main = "Forecasts from the Fitted ARMA (1,1) Model")
points(TSLA.test.bic, col = "firebrick4", pch = 19)

# Metrics for forecast accuracy (test set vs predicted mean)
accuracy(predfit.bic$mean, TSLA_test)

## 3.3 Comparison of forecast performance #####################################
forecast.performance.AR2 = accuracy(predfit.AR2$mean, TSLA_test)
forecast.performance.bic = accuracy(predfit.bic$mean, TSLA_test)
print(forecast.performance.AR2); print(forecast.performance.bic)

## 4 Non-Linear Models ########################################################
## 4.1 Examination of AR(2)+ARCH/GARCH Model ##################################
# Examination of Residuals
resid.AR2 = residuals(AR2)
par(mfrow = c(3, 1))
plot(resid.AR2, main = "Plot of AR(2) Residuals")
acf(resid.AR2, main = "ACF of AR(2) Residuals")
pacf(resid.AR2, main = "PACF of AR(2) Residuals")
par(mfrow = c(1, 1))

# Examination of Squared Residuals
sq.resid.AR2 = residuals(AR2)^2
par(mfrow = c(4, 1))
plot(sq.resid.AR2, main = "Plot of AR(2) Squared Residuals")
acf(sq.resid.AR2, main = "ACF of AR(2) Squared Residuals")
pacf(sq.resid.AR2, main = "PACF of AR(2) Squared Residuals")
McLeod.Li.test(y=resid.AR2, main = "McLeod-Li Test for AR(2) Model")
par(mfrow = c(1, 1))

## 4.2 Analysis of ARMA(1,1)+ARCH/GARCH Model #################################
# Examination of Residuals
resid.bic = residuals(auto.model.bic)
par(mfrow = c(3, 1))
plot(resid.bic, main = "Plot of ARMA(1,1) Residuals")
acf(resid.bic, main = "ACF of ARMA(1,1) Residuals")
pacf(resid.bic, main = "PACF of ARMA(1,1) Residuals")
par(mfrow = c(1, 1))

# Examination of Squared Residuals
sq.resid.bic = residuals(auto.model.bic)^2
par(mfrow = c(4, 1))
plot(sq.resid.bic, main = "Plot of ARMA(1,1) Squared Residuals")
acf(sq.resid.bic, main = "ACF of ARMA(1,1) Squared Residuals")
pacf(sq.resid.bic, main = "PACF of ARMA(1,1) Squared Residuals")
McLeod.Li.test(y=resid.bic, main = "McLeod-Li Test for ARMA(1,1) Model")
par(mfrow = c(1, 1))

## A Appendices ###############################################################
## Saving png for the document
png(
  filename = "coverage_probabilities.png",
  width  = 1600,
  height = 1000,
  res    = 150
)
par(
  mfrow = c(2, 1),
  mar = c(4, 4, 3, 2),
  oma = c(0, 0, 0, 0),
  cex = 1
)
dev.off()

# Note: This appendix was included to show how figures were exported for the report.
