#install.packages("python3")
#install.packages("reticulate")
library(reticulate)
np <- import("numpy")

#install.packages("kaos")
library(kaos)

mat <- np$load("sequenceFiles.npy", allow_pickle=TRUE)

for (i in 1:length(mat)) {
  tmp <- (mat[[i]])
  test = as.character(strsplit(tmp, "")[[1]])
  test.cgr = cgr(test,  res = 100)
  img <- cgr.plot(test.cgr, mode = "matrix")
  png(paste(i,".png"))
  plot(img)
  dev.off()
}