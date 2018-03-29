#Find appropriate learning rate and momentum rate for given number of hidden units for Tanh and quadratic loss function
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

#give initial lists
explist <- list()
accuracylist <- list()
vmselist <- list()
trmselist <- list()
stoplist <- list()

count=1

for (i in 1:length(momrates)){
    for (j in 1:length(rates)){
        #automatically uses softmax output
        m1 <- h2o.deeplearning(
            model_id="dl_model_first", 
            training_frame=train, 
            validation_frame=valid,   ## validation dataset: used for scoring and early stopping
            x=predictors,
            y=response,
            #hidden=c(200,200),       ## default: 2 hidden layers with 200 neurons each
            epochs=500,                ##  controls stopping times
            #variable_importances=T,    ## not enabled by default
            distribution = "multinomial", ##  we have categorical data
            standardize = T,  ## standardize the input data
            activation = "Tanh",  ## may choose Tanh or Rectifier, i.e. ReLU. Logistic sigmoid not included
            categorical_encoding="OneHotInternal",  ## this ensures the use of the 1 of c encoding with "OneHotInternal"
            loss = "Quadratic",  ## can be "Quadratic" or "CrossEntropy"
            adaptive_rate = F, ##turn adaptive rate adjustments off
            rate = rates[i], #the learning rate, start around 0.5, should be positive
            #to reduce number of hyper parameters we will do this so that 
            momentum_start = momrates[j], #the momentum at the start, overall must be 0<\alpha<1
            momentum_ramp = 1, #the number of samples overwhich the ramp occurs
            momentum_stable = momrates[j], #the stable momentum, should be again less than 1 and greater than 0
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
        Mat = h2o.confusionMatrix(m1, newdata=test, valid=FALSE)
        #accuracy on the training case
        accuracy = 1-tail(Mat$Error, n=1)
        
        #final mean squared error for the training and validation sets
        vmse <- h2o.mse(m1, valid=TRUE, train=FALSE)
        trmse <- h2o.mse(m1, valid=FALSE, train=TRUE)

        #now fit the convergence with a power law
        tr <- m1@model$scoring_history$training_rmse
        v <- m1@model$scoring_history$validation_rmse
        ep <- m1@model$scoring_history$epochs

        fit <- lm(log(ep)~log(tr))
        pow <- summary(fit)$coefficients[2,1]
        exp <- 1/pow
        
        #put everything in a list
        explist[[length(explist)+1]] <- exp
        accuracylist[[length(accuracylist)+1]] <- accuracy
        vmselist[[length(vmselist)+1]] <- vmse
        trmselist[[length(trmselist)+1]] <- trmse
        stoplist[[length(stoplist)+1]]  <- tail(ep, n=1)
        count
        count <- count+1
        }}


png(file = "ClassAcc_vRate_ClassAcc_Quad.jpg")
#plot the accuracy of everything
acc <- unlist(accuracylist)
Macc <- matrix(acc, nrow=length(momrates), ncol=length(rates), byrow = TRUE)
#set up the plot
plot(x=NULL,y=NULL, ylim=c(0.9,1.0), xlim=c(0,0.05), xlab = "Learning rate", ylab="Accuracy", main ="Classification Accuracy")
lines(rates, Macc[1,1:length(rates)], type = "o", col = "red")
lines(rates, Macc[2,1:length(rates)], type = "o", col = "blue")
lines(rates, Macc[3,1:length(rates)], type = "o", col = "black")
lines(rates, Macc[4,1:length(rates)], type = "o", col = "orange")
lines(rates, Macc[5,1:length(rates)], type = "o", col = "green")
legend(0.03, 1, c("Mom-rate=0.0", "Mom-rate=0.05","Mom-rate=0.1","Mom-rate=0.5","Mom-rate=0.9"),lwd=c(2.5,2.5),col=c("red","blue","black","orange","green"))
dev.off() 

png(file = "ClassAcc_vRate_exp_Quad.jpg")
#plot the convergence rate of it all
expon <- unlist(explist)
Mexp <- matrix(expon, nrow=length(momrates), ncol=length(rates), byrow = TRUE)
#set up the plot
plot(x=NULL,y=NULL, ylim=c(-1,0), xlim=c(0,0.05), xlab = "Learning rate", ylab="Exponent", main ="Convergence Rate Exponent for the Training MSE")
lines(rates, Mexp[1,1:length(rates)], type = "o", col = "red")
lines(rates, Mexp[2,1:length(rates)], type = "o", col = "blue")
lines(rates, Mexp[3,1:length(rates)], type = "o", col = "black")
lines(rates, Mexp[4,1:length(rates)], type = "o", col = "orange")
lines(rates, Mexp[5,1:length(rates)], type = "o", col = "green")
legend(.03, 0, c("Mom-rate=0.0", "Mom-rate=0.05","Mom-rate=0.1","Mom-rate=0.5","Mom-rate=0.9"),lwd=c(2.5,2.5),col=c("red","blue","black","orange","green"))
dev.off() 

png(file = "ClassAcc_vRate_MSE_Quad.jpg")
#plot the MSE for the validation case at the last iteration
vmse2 <- unlist(vmselist)
Mvmse <- matrix(vmse2, nrow=length(momrates), ncol=length(rates), byrow = TRUE)
#set up the plot
plot(x=NULL,y=NULL, ylim=c(0,0.1), xlim=c(0,0.05), xlab = "Learning rate", ylab="MSE", main ="MSE for the Validation Set")
lines(rates, Mvmse[1,1:length(rates)], type = "o", col = "red")
lines(rates, Mvmse[2,1:length(rates)], type = "o", col = "blue")
lines(rates, Mvmse[3,1:length(rates)], type = "o", col = "black")
lines(rates, Mvmse[4,1:length(rates)], type = "o", col = "orange")
lines(rates, Mvmse[5,1:length(rates)], type = "o", col = "green")
legend(0.03, 0.1, c("Mom-rate=0.0", "Mom-rate=0.05","Mom-rate=0.1","Mom-rate=0.5","Mom-rate=0.9"),lwd=c(2.5,2.5),col=c("red","blue","black","orange","green"))
dev.off()         

png(file = "ClassAcc_vRate_stop_Quad.jpg")
#plot the MSE for the validation case at the last iteration
stop <- unlist(stoplist)
Mstop <- matrix(stop, nrow=length(momrates), ncol=length(rates), byrow = TRUE)
#set up the plot
plot(x=NULL,y=NULL, ylim=c(0,500), xlim=c(0,0.05), xlab = "Learning rate", ylab="Stopping Time", main ="Early Stopping Time")
lines(rates, Mstop[1,1:length(rates)], type = "o", col = "red")
lines(rates, Mstop[2,1:length(rates)], type = "o", col = "blue")
lines(rates, Mstop[3,1:length(rates)], type = "o", col = "black")
lines(rates, Mstop[4,1:length(rates)], type = "o", col = "orange")
lines(rates, Mstop[5,1:length(rates)], type = "o", col = "green")
legend(0.03, 50,c("Mom-rate=0.0", "Mom-rate=0.05","Mom-rate=0.1","Mom-rate=0.5","Mom-rate=0.9"),lwd=c(2.5,2.5),col=c("red","blue","black","orange","green"))
dev.off()   