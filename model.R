
library(tidyverse)
library(keras)

xDrive <- readRDS( "data/xDrive.rds")
yDrive <- readRDS( "data/yDrive.rds")

## create new plot of one user
xDrive$train[2,,1:9] %>% ts() %>% plot()
xDrive$train[1,,1:9] %>% ts() %>% plot()



# the whole thing
model <- keras_model_sequential() %>% 
  layer_conv_1d(filters = 40, kernel_size = 30, strides = 2,
                activation = "relu", input_shape = c(1221, 11)) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 40, kernel_size = 10, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 40, kernel_size = 10, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = "sigmoid") %>%
  layer_dense(units = 3, activation = "softmax")

model %>% compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = c("accuracy"))
history <- model %>% fit(xDrive$train, yDrive$train,
                         epochs = 15, 
                         batch_size = 128, 
                         validation_split = 0.3,
                         verbose = 1)

model %>% 
  evaluate(xDrive$test, yDrive$test)





# the whole thing
model <- keras_model_sequential() %>% 
  layer_conv_1d(filters = 5, kernel_size = 10, strides = 2,
                activation = "relu", input_shape = c(1221, 11)) %>%
  layer_max_pooling_1d(pool_size = 4) %>%
  layer_conv_1d(filters = 5, kernel_size = 10, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 4) %>%
  layer_conv_1d(filters = 5, kernel_size = 10, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 4) %>%
  layer_flatten() %>%
  layer_dense(units = 10, activation = "sigmoid") %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = "sigmoid") %>%
  layer_dense(units = 3, activation = "softmax")

model %>% compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = c("accuracy"))
history <- model %>% fit(xDrive$train, yDrive$train,
                         epochs = 20, 
                         batch_size = 128, 
                         validation_split = 0.3,
                         verbose = 1)

model %>% 
  evaluate(xDrive$test, yDrive$test)



### LSTM model
model <- keras_model_sequential() %>% 
  layer_lstm(units = 24, input_shape = c(1221, 11)) %>% 
  layer_dense(units = 3, activation = 'sigmoid')

summary(model)
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)
history <- model %>% 
  fit(xDrive$train, yDrive$train,
                         epochs = 20, 
                         batch_size = 128, 
                         validation_split = 0.3,
                         verbose = 1)
plot(history)
model %>% 
  evaluate(xDrive$test, yDrive$test)





# Make a new model

model <- keras_model_sequential()

# 1D because its a time series
# Conv layer
model %>%
  layer_conv_1d(filters = 40, 
                kernel_size = 40, 
                strides = 2,
                activation = "relu", 
                input_shape = c(1221, 11))
model

# Max pool
model %>%
  layer_max_pooling_1d(pool_size = 2)
model

# Flatten
model %>%
  layer_flatten()
model

# Finish
model %>%
  layer_dense(units = 100, activation = "sigmoid") %>%
  layer_dense(units = 3, activation = "softmax")
model



model %>% 
  compile(loss = "categorical_crossentropy", 
          optimizer = "adam", 
          metrics = c("accuracy"))

history <- model %>% fit(xDrive$train, yDrive$train,
                         epochs = 15, 
                         validation_split = 0.3,
                         verbose = 1)


model %>% 
  evaluate(xDrive$test, yDrive$test)


