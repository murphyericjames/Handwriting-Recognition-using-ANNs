#Find number of hidden units and layers for Quadratic loss function 
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
#set layers as 1,2,5,8,10
layers <- c(1,2,5,8,10)
#set units per layer as 60,50,40,30,20,10
units <- c(60,50,40,30,20,10)
hiddenlist <- list()
for (i in 1:length(layers)){
    for (j in 1:length(units)){
        hiddenlist[[length(hiddenlist)+1]] <- rep_len(units[j],layers[i])
    }}
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

for (i in 1:length(layers)){
    for (j in 1:length(units)){
        #automatically uses softmax output
        m1 <- h2o.deeplearning(
            model_id="dl_model_first", 
            training_frame=train, 
            validation_frame=valid,   ## validation dataset: used for scoring and early stopping
            x=predictors,
            y=response,
            #hidden=c(200,200),       ## default: 2 hidden layers with 200 neurons each
            epochs=1000,                ##  controls stopping times
            #variable_importances=T,    ## not enabled by default
            distribution = "multinomial", ##  we have categorical data
            standardize = T,  ## standardize the input data
            activation = "Tanh",  ## may choose Tanh or Rectifier, i.e. ReLU. Logistic sigmoid not included
            categorical_encoding="OneHotInternal",  ## this ensures the use of the 1 of c encoding with "OneHotInternal"
            loss = "Quadratic",  ## can be "Quadratic" or "CrossEntropy"
            adaptive_rate = F, ##turn adaptive rate adjustments off
            rate = 0.01, #the learning rate, start around 0.5, should be positive
            #to reduce number of hyper parameters we will do this so that 
            momentum_start = 0.0, #the momentum at the start, overall must be 0<\alpha<1
            momentum_ramp = 1, #the number of samples overwhich the ramp occurs
            momentum_stable = 0.0, #the stable momentum, should be again less than 1 and greater than 0
            #train_samples_per_iteration = 0, #number of training samples per mapreduce iteraton, 
            #special vals 0 one epoch, -1 all available data, -2 autotuning
            #want to disable early stopping
            classification_stop = -1,
            regression_stop = -1,
            overwrite_with_best_model=FALSE,  #don't save the best model
            hidden=unlist(hiddenlist[count])#c(50,50) ## hidden layers c(100,100) 2 hidden layers with 100 neurons each
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

png(file = "ClassAcc_vHidden_ClassAcc_Quad.jpg")
#plot the accuracy of everything
acc <- unlist(accuracylist)
Macc <- matrix(acc, nrow=length(layers), ncol=length(units), byrow = TRUE)
#set up the plot
plot(x=NULL,y=NULL, ylim=c(0.85,0.95), xlim=c(10,60), xlab = "Hidden units", ylab="Accuracy", main ="Classification Accuracy")
lines(units, Macc[1,1:length(units)], type = "o", col = "red")
lines(units, Macc[2,1:length(units)], type = "o", col = "blue")
lines(units, Macc[3,1:length(units)], type = "o", col = "black")
lines(units, Macc[4,1:length(units)], type = "o", col = "orange")
lines(units, Macc[5,1:length(units)], type = "o", col = "green")
legend(40, 0.89, c("1-layer", "2-layer","5-layer","8-layer","10-layer"),lwd=c(2.5,2.5),col=c("red","blue","black","orange","green"))
dev.off()

png(file = "ClassAcc_vHidden_ConvExp_Quad.jpg")
#plot the convergence rate of it all
expon <- unlist(explist)
Mexp <- matrix(expon, nrow=length(layers), ncol=length(units), byrow = TRUE)
#set up the plot
plot(x=NULL,y=NULL, ylim=c(-3,0), xlim=c(10,60), xlab = "Hidden units", ylab="Exponent", main ="Convergence Rate Exponent for the Training MSE")
lines(units, Mexp[1,1:length(units)], type = "o", col = "red")
lines(units, Mexp[2,1:length(units)], type = "o", col = "blue")
lines(units, Mexp[3,1:length(units)], type = "o", col = "black")
lines(units, Mexp[4,1:length(units)], type = "o", col = "orange")
lines(units, Mexp[5,1:length(units)], type = "o", col = "green")
legend(40, -1.5, c("1-layer", "2-layer","5-layer","8-layer","10-layer"),lwd=c(2.5,2.5),col=c("red","blue","black","orange","green"))
dev.off()       

png(file = "ClassAcc_vHidden_MSEval_Quad.jpg")
#plot the MSE for the validation case at the last iteration
vmse2 <- unlist(vmselist)
Mvmse <- matrix(vmse2, nrow=length(layers), ncol=length(units), byrow = TRUE)
#set up the plot
plot(x=NULL,y=NULL, ylim=c(0,0.15), xlim=c(10,60), xlab = "Hidden units", ylab="MSE", main ="MSE for the Validation Set")
lines(units, Mvmse[1,1:length(units)], type = "o", col = "red")
lines(units, Mvmse[2,1:length(units)], type = "o", col = "blue")
lines(units, Mvmse[3,1:length(units)], type = "o", col = "black")
lines(units, Mvmse[4,1:length(units)], type = "o", col = "orange")
lines(units, Mvmse[5,1:length(units)], type = "o", col = "green")
legend(40, 0.15, c("1-layer", "2-layer","5-layer","8-layer","10-layer"),lwd=c(2.5,2.5),col=c("red","blue","black","orange","green"))
dev.off()    

png(file = "ClassAcc_vHidden_Stop_Quad.jpg")
#plot the MSE for the validation case at the last iteration
stop <- unlist(stoplist)
Mstop <- matrix(stop, nrow=length(layers), ncol=length(units), byrow = TRUE)
#set up the plot
plot(x=NULL,y=NULL, ylim=c(1,1000), xlim=c(10,60), log="xy", xlab = "Hidden units", ylab="Stopping Epoch", main ="Early Stopping Time")
lines(units, Mstop[1,1:length(units)], type = "o", col = "red")
lines(units, Mstop[2,1:length(units)], type = "o", col = "blue")
lines(units, Mstop[3,1:length(units)], type = "o", col = "black")
lines(units, Mstop[4,1:length(units)], type = "o", col = "orange")
lines(units, Mstop[5,1:length(units)], type = "o", col = "green")
legend(20, 50, c("1-layer", "2-layer","5-layer","8-layer","10-layer"),lwd=c(2.5,2.5),col=c("red","blue","black","orange","green"))
dev.off()  