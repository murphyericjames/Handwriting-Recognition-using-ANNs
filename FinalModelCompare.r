#compare all three final models
library(h2o)
localH2O<- h2o.init()

#this function does not actually exist
#h2o.addFunction(localH2O, function(x) { 1/(1+exp(-x)) }, "Logsig")

#Import the data from the website repository, first name the path
train_file <- "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
test_file <- "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"

#give path to the importFile command in R
train1 <- h2o.importFile(train_file)
test <- h2o.importFile(test_file)

#get summary of data
#summary(train)
#summary(test)

#get a vector of the number of instances and the length of the feature vectors
dtrain <- dim(train1)
dtest <- dim(test)

#split training data at random for a selection to train on and a selection to validate hyperparameters
splits <- h2o.splitFrame(train1, c(0.8), seed=1385)
train <- h2o.assign(splits[[1]], "train.hex")
valid <- h2o.assign(splits[[2]], "valid.hex")

#can do some initial data inspection using, C65 is the labeled category
#par(mfrow=c(1,1))
#plot(h2o.tabulate(train, "C40", "C65"))
#plot(h2o.tabulate(train, "C30", "C65"))


#treat vectors as integer categories(factors), and thus not factors
train <- as.factor(train)
valid <- as.factor(valid)
test <- as.factor(test)

#set the target and feature(predictor) vectors
response <- "C65"
predictors <- setdiff(names(train1), response)
predictors

#set parameters for the hidden layers search, with low learning rate, e.g. the 0.01
#set number of  momentum rates to look at
momrates <- c(0.0, 0.05, 0.1, 0.5, 0.9)
#set number of rates to look at
rates <- c(0.001,0.005,0.01,0.02,0.05)

#give a string index for all of these lists
#choose hidden layers, by looping over a set of them
#extract convergence rate and accuracy
#do study on learning rate for both tanh, and Rectifier for the hyperparameters


count=1

#the tanh activation function with cross-entropy loss function
        #automatically uses softmax output
        m1 <- h2o.deeplearning(
            model_id="dl_model_first", 
            training_frame=train, 
            validation_frame=valid,   ## validation dataset: used for scoring and early stopping
            x=predictors,
            y=response,
            #hidden=c(200,200),       ## default: 2 hidden layers with 200 neurons each
            epochs=500,#000,                ##  controls stopping times
            #variable_importances=T,    ## not enabled by default
            distribution = "multinomial", ##  we have categorical data
            standardize = T,  ## standardize the input data
            activation = "Tanh",  ## may choose Tanh or Rectifier, i.e. ReLU. Logistic sigmoid not included
            categorical_encoding="OneHotInternal",  ## this ensures the use of the 1 of c encoding with "OneHotInternal"
            loss = "CrossEntropy",  ## can be "Quadratic" or "CrossEntropy"
            adaptive_rate = F, ##turn adaptive rate adjustments off
            rate = 0.02, #the learning rate, start around 0.5, should be positive
            #to reduce number of hyper parameters we will do this so that 
            momentum_start = 0.5, #the momentum at the start, overall must be 0<\alpha<1
            momentum_ramp = 1, #the number of samples overwhich the ramp occurs
            momentum_stable = 0.5, #the stable momentum, should be again less than 1 and greater than 0
            #train_samples_per_iteration = 0, #number of training samples per mapreduce iteraton, 
            #special vals 0 one epoch, -1 all available data, -2 autotuning
            #want to disable early stopping
            classification_stop = -1,
            regression_stop = -1,
            overwrite_with_best_model=FALSE,  #don't save the best model
            #change to best hidden list
            hidden=c(50,50)#unlist(hiddenlist[count])#c(50,50) ## hidden layers c(100,100) 2 hidden layers with 100 neurons each
        )
        
        #find the convergence statistics for the 
        #find performance statistics i.e. accuracy for the training set
        #m2 = h2o.performance(m1, newdata=test, train=FALSE, valid=FALSE, xval=FALSE)
        Mat = h2o.confusionMatrix(m1, newdata=test, valid=FALSE)
        Mat
        #accuracy on the training case
        accuracyscaled = 1-tail(Mat$Error, n=1)
        accuracyscaled


        #now fit the convergence with a power law
        tr <- m1@model$scoring_history$training_rmse
        v <- m1@model$scoring_history$validation_rmse
        ep <- m1@model$scoring_history$epochs

        fit <- lm(log(ep)~log(tr))
        pow1 <- summary(fit)$coefficients[2,1]
        exp1 <- 1/pow1
        print("Convergence Exponent Cross-Entropy loss")
        exp1



        #the tanh activation function with Quadratic loss function 
        #automatically uses softmax output
        m2 <- h2o.deeplearning(
            model_id="dl_model_first", 
            training_frame=train, 
            validation_frame=valid,   ## validation dataset: used for scoring and early stopping
            x=predictors,
            y=response,
            #hidden=c(200,200),       ## default: 2 hidden layers with 200 neurons each
            epochs=500,#000,                ##  controls stopping times
            #variable_importances=T,    ## not enabled by default
            distribution = "multinomial", ##  we have categorical data
            standardize = F,#T,  ## standardize the input data
            activation = "Tanh",  ## may choose Tanh or Rectifier, i.e. ReLU. Logistic sigmoid not included
            categorical_encoding="OneHotInternal",  ## this ensures the use of the 1 of c encoding with "OneHotInternal"
            loss = "Quadratic",  ## can be "Quadratic" or "CrossEntropy"
            adaptive_rate = F, ##turn adaptive rate adjustments off
            rate = 0.02, #the learning rate, start around 0.5, should be positive
            #to reduce number of hyper parameters we will do this so that 
            momentum_start = 0.9, #the momentum at the start, overall must be 0<\alpha<1
            momentum_ramp = 1, #the number of samples overwhich the ramp occurs
            momentum_stable = 0.9, #the stable momentum, should be again less than 1 and greater than 0
            #train_samples_per_iteration = 0, #number of training samples per mapreduce iteraton, 
            #special vals 0 one epoch, -1 all available data, -2 autotuning
            #want to disable early stopping
            classification_stop = -1,
            regression_stop = -1,
            overwrite_with_best_model=FALSE,  #don't save the best model
            #change to best hidden list
            hidden=c(40,40)#unlist(hiddenlist[count])#c(50,50) ## hidden layers c(100,100) 2 hidden layers with 100 neurons each
        )
        
        #find the convergence statistics for the 
        #find performance statistics i.e. accuracy for the training set
        #m2 = h2o.performance(m1, newdata=test, train=FALSE, valid=FALSE, xval=FALSE)
        Mat2 = h2o.confusionMatrix(m2, newdata=test, valid=FALSE)
        Mat2
        #accuracy on the training case
        accuracy2 = 1-tail(Mat2$Error, n=1)
        print("Accuracy Quadratic loss")
        accuracy2

        #now fit the convergence with a power law
        tr <- m2@model$scoring_history$training_rmse
        v <- m2@model$scoring_history$validation_rmse
        ep <- m2@model$scoring_history$epochs

        fit <- lm(log(ep)~log(tr))
        pow2 <- summary(fit)$coefficients[2,1]
        exp2 <- 1/pow2
        print("Convergence Exponent Quadratic loss")
        exp2


#the ReLU activation function with Cross-Entropy loss function 
        #automatically uses softmax output
        m3 <- h2o.deeplearning(
            model_id="dl_model_first", 
            training_frame=train, 
            validation_frame=valid,   ## validation dataset: used for scoring and early stopping
            x=predictors,
            y=response,
            #hidden=c(200,200),       ## default: 2 hidden layers with 200 neurons each
            epochs=500,#000,                ##  controls stopping times
            #variable_importances=T,    ## not enabled by default
            distribution = "multinomial", ##  we have categorical data
            standardize = T,  ## standardize the input data
            activation = "Rectifier",  ## may choose Tanh or Rectifier, i.e. ReLU. Logistic sigmoid not included
            categorical_encoding="OneHotInternal",  ## this ensures the use of the 1 of c encoding with "OneHotInternal"
            loss = "CrossEntropy",  ## can be "Quadratic" or "CrossEntropy"
            adaptive_rate = F, ##turn adaptive rate adjustments off
            rate = 0.005, #the learning rate, start around 0.5, should be positive
            #to reduce number of hyper parameters we will do this so that 
            momentum_start = 0.1, #the momentum at the start, overall must be 0<\alpha<1
            momentum_ramp = 1, #the number of samples overwhich the ramp occurs
            momentum_stable = 0.1, #the stable momentum, should be again less than 1 and greater than 0
            #train_samples_per_iteration = 0, #number of training samples per mapreduce iteraton, 
            #special vals 0 one epoch, -1 all available data, -2 autotuning
            #want to disable early stopping
            classification_stop = -1,
            regression_stop = -1,
            overwrite_with_best_model=FALSE,  #don't save the best model
            #change to best hidden list
            hidden=c(50,50)#unlist(hiddenlist[count])#c(50,50) ## hidden layers c(100,100) 2 hidden layers with 100 neurons each
        )
        
        #find the convergence statistics for the 
        #find performance statistics i.e. accuracy for the training set
        #m2 = h2o.performance(m1, newdata=test, train=FALSE, valid=FALSE, xval=FALSE)
        Mat3 = h2o.confusionMatrix(m3, newdata=test, valid=FALSE)
        Mat3
        #accuracy on the training case
        accuracy3 = 1-tail(Mat3$Error, n=1)
        print("accuracy ReLU")
        accuracy3

        #now fit the convergence with a power law
        tr <- m3@model$scoring_history$training_rmse
        v <- m3@model$scoring_history$validation_rmse
        ep <- m3@model$scoring_history$epochs

        fit <- lm(log(ep)~log(tr))
        pow3 <- summary(fit)$coefficients[2,1]
        exp3 <- 1/pow3
        print("Convergence Exponent ReLU")
        exp3