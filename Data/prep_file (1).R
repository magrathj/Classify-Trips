library(tidyverse)
library(keras)


driving <- readRDS( "data/driving (5).rds")

driving_x <- driving[,,1:11] %>%
  apply(c(1, 3), scale) %>% # scaling within each series so no issue with train/test
  aperm(c(2,1,3))           # undo apply's ridiculous transpose

driving_x[is.nan(driving_x)] <- 0
print(driving_x[,1,])



dimnames(driving_x) <- list(NULL, NULL, c("X_Accel", "Y_Accel", "Z_Accel", "Vehicle Spd (km/h)", "Throttle Pos (%)", "velocity", "Odometer", "YawRate", "PitchRate", "Vehicle Acceleration HEM (m/s/s)", "VSP"))

driving_labels <- driving[,1,12] %>%
  as.integer()

drivingData <- list(x = driving_x, 
                    labels = driving_labels)
saveRDS(drivingData, "/cloud/project/data/drivingData2.rds")

drivingData <- readRDS( "data/drivingData2.rds")
drivingData %>% 
  head()


for(i in 1:72){
  drivingData$x[i,,1:9] %>% ts() %>% plot()
}

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


