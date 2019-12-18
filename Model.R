install.packages(car)
library(car)
install.packages("boot")
library(boot)
install.packages("confidence")
library('confidence')

getwd()
setwd("C:/Users/CosmicDust/Documents/Statistics/Final Project")
open_train = read.csv("open.csv")


#--models for Open Season

model_2 = lm(Recreation.Visits ~ LowestTemperature.F. + HighestTemperature.F. +
               WarmestMinimumTemperature.F.+ColdestMaximumTemperature.F.+
               AverageMinimumTemperature.F.+AverageMaximumTemperature.F.+ 
               MeanTemperature.F.+TotalPrecipitation.In.+TotalSnowfall.In.+
               Max.24hrPrecipitation.In.+Max.24hrSnowfall.In., data = open_train)
summary(model_2)

model_2.1 = lm(log(Recreation.Visits) ~ LowestTemperature.F. + 
                 HighestTemperature.F. +WarmestMinimumTemperature.F.+
                 ColdestMaximumTemperature.F.+AverageMinimumTemperature.F.+
                 AverageMaximumTemperature.F.+ MeanTemperature.F.+sqrt(TotalSnowfall.In.)+
                 sqrt(Max.24hrSnowfall.In.), data = open_train)
summary(model_2.1)
vif(model_2.1)
anova(model_2.1)
AIC(model_2.1)
BIC(model_2.1)


#model_2.2 good
model_2.2 = lm(log(Recreation.Visits) ~ LowestTemperature.F. +
                 WarmestMinimumTemperature.F.+ColdestMaximumTemperature.F.+
                 AverageMinimumTemperature.F.+AverageMaximumTemperature.F.+ 
                 MeanTemperature.F.+sqrt(Max.24hrSnowfall.In.), data = open_train)
summary(model_2.2)
vif(model_2.2)
anova(model_2.2)
AIC(model_2.2)
BIC(model_2.2)

#model 2.3: 
model_2.3 = lm(log(Recreation.Visits) ~ LowestTemperature.F. +WarmestMinimumTemperature.F.+ColdestMaximumTemperature.F.+AverageMaximumTemperature.F.+ sqrt(Max.24hrSnowfall.In.), data = open_train)
summary(model_2.3)
vif(model_2.3)
anova(model_2.3)
AIC(model_2.3)
BIC(model_2.3)


# model_2.4:based on vif
model_2.4 = lm(log(Recreation.Visits) ~ WarmestMinimumTemperature.F.+ sqrt(Max.24hrSnowfall.In.), data = open_train)
summary(model_2.4)
vif(model_2.4)
anova(model_2.4)
AIC(model_2.4)
BIC(model_2.4)

par(mfrow = c(2, 2))
plot(model_2.4)



# ---------------------------- Final Model for open Season -------------------------------

# model_3 :very good: eliminating multi collinearity-vif (below 5) ,significance values, 
# acceptable rsquare(0.8), high -error(0.69),  high ----(bic-799.4, aic-799)

model_3 = glm(log(Recreation.Visits) ~ (MeanTemperature.F.) + 
                log(AverageMinimumTemperature.F.) + log(HighestTemperature.F.) + 
                Max.24hrPrecipitation.In., data = open_train)


summary(model_3)
vif(model_3)
anova(model_3)
AIC(model_3)
BIC(model_3)


par(mfrow = c(2, 2))
plot(model_3)

# Cross Validation
set.seed(6)
cv_results=cv.glm(open_train, model_3, K=5)
cv_results$delta #smaller delta the better fit

#---------------------------------------------------------------------------------------------

