library(readr)
library(xgboost)

train <- read_csv("benchmark20_train.csv")
test  <- read_csv("benchmark20_test.csv")
test[is.na(test)] <- 1 # NA's are in 'Open'

seed <- 1
train$logSales <- log(train$Sales+1)

for (f in names(train)) {
  if (class(train[[f]])=="character") {
    levels <- unique(train[[f]])
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
  }
}

for (f in names(test)) {
  if (class(test[[f]])=="character") {
    levels <- unique(test[[f]])
    test[[f]] <- as.integer(factor(test[[f]], levels=levels))
  }
}



feature <- list()
feature[[1]] <- names(train[,c(7,1,2,3,5,8,9,10,11,12,13,14,15,16,23,22,25)])
feature[[2]] <- names(train[,c(1,3,5,6,7,8,9,10,13,17,19,20,21,22,23,24,25)])
feature[[3]] <- names(train[,-c(2,4,11,12,14,15,16,18,33)])

RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))-1
  epreds <- exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

RMPSE.obj <- function (preds,dtrain) {
  labels <- getinfo(dtrain, "label")
  grad <- -1/labels+preds/(labels**2)
  hess <- 1/(labels**2)
  return(list(grad=grad, hess=hess))
}

param <- list(  objective           = RMPSE.obj,  
                booster             = "gbtree",
                eta                 = 0.03,
                max_depth           = 15,
                subsample           = 0.7,
                colsample_bytree    = 0.7,
                nthread             = 4
)

xgb_fill <- vector(mode='numeric', length = nrow(train))
xgb_fill_1 <- vector(mode='numeric', length = nrow(train))

rounds <- list()
rounds[[1]] <- list(2100, 125)
rounds[[2]] <- list(3732, 611)
rounds[[3]] <- list(2979, 740)

for (f in 1:3) {
  feature.names <- feature[[f]]
  rnd <- rounds[[f]][[1]]
  std <- rounds[[f]][[2]]

    for (i in 0:4) {
    
    cat('\n', 'feature.name =',f, 'fold =',i+1,'\n')
    timestamp()
    train_idx <- read_csv(file=paste0('train5F_',i,'.csv'))$idx + 1
    fill_idx <- read_csv(file=paste0('fill5F_',i,'.csv'))$idx + 1
    dtrain <- xgb.DMatrix(data=data.matrix(train[train_idx,feature.names]), label=train$logSales[train_idx])
    dfill <- xgb.DMatrix(data=data.matrix(train[fill_idx,feature.names]), label=train$logSales[fill_idx])
    watchlist <-list(val=dfill, train=dtrain)
    
    set.seed(seed)
    clf <- xgb.train(   params              = param, 
                        data                = dtrain, 
                        nrounds             = rnd, #rnd
                        feval               = RMPSE,
                        watchlist           = watchlist,
                        verbose             = 1,
                        maximize            = FALSE,
                        print.every.n       = 1000
                        )
    
    pred   <- exp(predict(clf, data.matrix(train[fill_idx,feature.names]))) -1
    pred_1 <- exp(predict(clf, data.matrix(train[fill_idx,feature.names]), ntreelimit=rnd-std)) -1  # rnd-std
    xgb_fill[fill_idx] <- pred
    xgb_fill_1[fill_idx] <- pred_1
  }


df <- data.frame(id=1:nrow(train), xgb_fill=xgb_fill)
write_csv(df, path = paste0('xgb_fill_feature_', f, '.csv'))

df_1 <- data.frame(id=1:nrow(train), xgb_fill=xgb_fill_1)
write_csv(df_1, path=paste0('xgb_fill_feature_', f, '_1.csv'))

}



for (f in 1:3) {
  
  feature.names <- feature[[f]]
  rnd <- rounds[[f]][[1]]
  std <- rounds[[f]][[2]]  
  
  dtrain <- xgb.DMatrix(data=data.matrix(train[,feature.names]), label=train$logSales)
  dtest  <- xgb.DMatrix(data=data.matrix(test[,feature.names]))
  
  set.seed(seed)
  clf1 <- xgb.train(  params              = param, 
                      data                = dtrain, 
                      nrounds             = rnd, # rnd
                      feval               = RMPSE,
                      verbose             = 1,
                      maximize            = FALSE,
                      print.every.n       = 1000
  )
  
  pred <- exp(predict(clf1, newdata=dtest)) -1
  df1 <- data.frame(id= test$Id , xgb_test=pred)
  write_csv(df1, path=paste0('xgb_test_feature_',f,'.csv'))
  
  pred2 <- exp(predict(clf1, newdata=dtest, ntreelimit=rnd-std)) -1 # rnd - std
  df2 <- data.frame(id= test$Id , xgb_test_1=pred2)
  write_csv(df2, path=paste0('xgb_test_feature_',f,'_1.csv'))
}
