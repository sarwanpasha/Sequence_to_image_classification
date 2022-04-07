library(ggplot2)
#install.packages("reshape2")
library(reshape)
library(reticulate)
np <- import("numpy")
library(stringi)
library(stringr)

distr.pts = function(n,
                     r,
                     plot = F){
  
  #get coordinates for a regular polygon
  x = vector("double", n)
  y = vector("double", n)
  for (i in 1:n){
    x[i] = r*sinpi((2*i+1)/n)
    y[i] = r*cospi((2*i+1)/n)
  }
  
  #generates a plot if required
  if (plot) {plot(x, y, pch = 20)}
  
  #return coordinates
  return(xy.coords(x, y))
}

cgr = function(data,
               seq.base = row.names(table(data)),
               sf = F,
               res = 100) {
  
  r = 1
  if(is.character(seq.base)&&length(seq.base)==1){
    if(seq.base == "digits"){
      seq.base =c(0:9)
    }
    else if(seq.base == "AMINO"){
      seq.base=c("A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R",
                 "S","T","V","W","Y")
    }
    else if(seq.base == "amino"){
      seq.base=c("a","c","d","e","f","g","h","i","k","l","m","n","p","q","r",
                 "s","t","v","w","y")
    }
    
    else if (seq.base == "DNA"){
      seq.base= c("A","G","T","C")
    }
    
    else if (seq.base == "dna"){
      seq.base= c("a","g","t","c")
    }
    else if (seq.base == "LETTERS"){
      seq.base=LETTERS
    }
    
    else if (seq.base == "letters"){
      seq.base=letters
    }
    
  }
  
  stopifnot(
    length(seq.base) >= length(table(data)),
    all(row.names(table(data)) %in% seq.base),
    sf <= 1,
    sf >= 0,
    res >= 1)
  
  base.num = length(seq.base)
  
  if(base.num==4){
    x=c(1,-1,-1,1)
    y=c(-1,-1,1,1)
    base.coord = xy.coords(x, y)
  }
  #calculate corner coordinates for the base
  else{
    
    base.coord = distr.pts(base.num, r)
  }
  
  if (!sf) {sf =  1- (sinpi(1/base.num)/ (sinpi(1 / base.num) + sinpi (
    1/base.num + 2 * (floor (base.num/4) /base.num))))}
  
  #data frame for easy access
  base = data.frame(x = base.coord$x,
                    y = base.coord$y,
                    row.names = seq.base)
  
  #get the length of data
  data.length = length(data)
  
  x = vector("double", data.length)
  y = vector("double", data.length)
  A = matrix(data = 0, ncol = res, nrow = res)
  pt = vector("double", 2)

  for (i in seq(1, data.length, by=1)){
    for (j in i:(i+8)){
      pt = pt + (unlist(base[data[j],]) - pt) * sf
      x[j] = pt[1]
      y[j] = pt[2]
      x.matrix = ceiling((x[j]+r ) * res/(2*r))
      y.matrix = ceiling((y[j]+r ) * res/(2*r))
      A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + 1
    }
  }
  
  #return matrix, coordinates, scaling factor, resolution
  return(list(matrix = A,
              x = x,
              y = y,
              scaling_factor = sf,
              resolution = res,
              base = base))
}

build_minimizer = function(s,k,m){
  L = nchar(s)
  rev <- stri_reverse(s)
  minimizers <- c()
  k_mers_final <- c()
  for( i in 1:(L-k+1) ){ 
    sub_f <- substring(s, i, (i+k-1)) 
    sub_r <- stri_reverse(sub_f)     
    min = "ZZZZZZZZZZZZZ"
    for ( j in 1:(k-m+1) ){
      sub2 <- substring(sub_f, j, (j+m-1)) 
      if(sub2 < min){
        min <- sub2
      }
      sub2 <- substring(sub_r, j, (j+m-1))
      if(sub2 < min){
        min <- sub2
      }
    }
    minimizers <- c(minimizers, min)
    k_mers_final <- c(k_mers_final, sub_f)
  }
  return(minimizers)
}

mat <- np$load("sequenceFiles.npy", allow_pickle=TRUE)

for (i in 1:length(mat)) {
  s <- (mat[[i]])
  minis = build_minimizer(s,9,3)
  min_data = str_flatten(minis)
  test = as.character(strsplit(min_data, "")[[1]])
  test_cgr = cgr(test,  res = 1)
  png(paste(i,".png"))
  plot(x=test_cgr$x, y=test_cgr$y, xlab='', ylab='', xaxt="n", yaxt="n")
  dev.off()
}
