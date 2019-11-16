library(tidyverse)
library(keras)

drivingData <- readRDS( "data/drivingData.rds")
drivingData %>% 
  head()


set.seed(19)
m <- nrow(drivingData$x)

print(m)

# generate random indicies
indices <- sample(1:m, m)

indTrain <- indices[1:floor(m*0.6)]
indVal <- indices[ceiling(m*0.6):floor(m*0.8)]
indTest <- indices[ceiling(m*0.8):m]

drivingData$y <- to_categorical(drivingData$labels - 1)

xDrive <- list(train = drivingData$x[indTrain, ,],
               val = drivingData$x[indVal, ,],
               test = drivingData$x[indTest, ,])

yDrive <- list(train = drivingData$y[indTrain, ],
               val = drivingData$y[indVal, ],
               test = drivingData$y[indTest, ])

saveRDS(xDrive, "data/xDrive.rds")
saveRDS(yDrive, "data/yDrive.rds")

View(drivingData)


