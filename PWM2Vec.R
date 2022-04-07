library(ggplot2)
#install.packages("reshape2")
library(reshape)
library(reticulate)
np <- import("numpy")

get_kmers = function(seq, kmer){
  seq = seq
  kmer = kmer
  L = nchar(seq)
  end_ind = L-kmer+1
  k_mers_final <- c()
  for (i in 1:end_ind){
    sub_f <- substring(seq, i, i+8) 
    k_mers_final <- c(k_mers_final, sub_f)
  }
  return(k_mers_final)
}

pwm = function(k_mers_final){
  a_val = c(0,0,0,0,0,0,0,0,0)
  b_val = c(0,0,0,0,0,0,0,0,0)
  c_val = c(0,0,0,0,0,0,0,0,0)
  d_val = c(0,0,0,0,0,0,0,0,0)
  e_val = c(0,0,0,0,0,0,0,0,0)
  f_val = c(0,0,0,0,0,0,0,0,0)
  g_val = c(0,0,0,0,0,0,0,0,0)
  h_val = c(0,0,0,0,0,0,0,0,0)
  i_val = c(0,0,0,0,0,0,0,0,0)
  j_val = c(0,0,0,0,0,0,0,0,0)
  k_val = c(0,0,0,0,0,0,0,0,0)
  l_val = c(0,0,0,0,0,0,0,0,0)
  m_val = c(0,0,0,0,0,0,0,0,0)
  n_val = c(0,0,0,0,0,0,0,0,0)
  p_val = c(0,0,0,0,0,0,0,0,0)
  q_val = c(0,0,0,0,0,0,0,0,0)
  r_val = c(0,0,0,0,0,0,0,0,0)
  s_val = c(0,0,0,0,0,0,0,0,0)
  t_val = c(0,0,0,0,0,0,0,0,0)
  v_val = c(0,0,0,0,0,0,0,0,0)
  w_val = c(0,0,0,0,0,0,0,0,0)
  x_val = c(0,0,0,0,0,0,0,0,0)
  y_val = c(0,0,0,0,0,0,0,0,0)
  z_val = c(0,0,0,0,0,0,0,0,0)
  
  count_lines = 0 
  for ( ii in 1:(length(k_mers_final)) ){
    line = k_mers_final[ii]
    count_lines = count_lines +1
    
    for (i in 1:nchar(line)) {
      if (substring(line,i,i) == 'A') {
        a_val[i] = a_val[i]+1
      }
      else if (substring(line,i,i) == 'B') {
        b_val[i] = b_val[i]+1
      }
      else if (substring(line,i,i) == 'C'){
        c_val[i] = c_val[i]+1
      }
      else if (substring(line,i,i) == 'D') {
        d_val[i] = d_val[i]+1
      }
      else if (substring(line,i,i) == 'E') {
        e_val[i] = e_val[i]+1
      }
      else if (substring(line,i,i) == 'F') {
        f_val[i] = f_val[i]+1
      }
      else if (substring(line,i,i) == 'G') {
        g_val[i] = g_val[i]+1 
      }
      else if (substring(line,i,i) == 'H') {
        h_val[i] = h_val[i]+1
      }
      else if (substring(line,i,i) == 'I') {
        i_val[i] = i_val[i]+1
      }
      else if (substring(line,i,i) == 'J') {
        j_val[i] = j_val[i]+1
      }
      else if (substring(line,i,i) == 'K') {
        k_val[i] = k_val[i]+1
      }
      else if (substring(line,i,i) == 'L') {
        l_val[i] = l_val[i]+1
      }
      else if (substring(line,i,i) == 'M') {
        m_val[i] = m_val[i]+1
      }
      else if (substring(line,i,i) == 'N') {
        n_val[i] = n_val[i]+1
      }
      else if (substring(line,i,i) == 'P') {
        p_val[i] = p_val[i]+1
      }
      else if (substring(line,i,i) == 'Q') {
        q_val[i] = q_val[i]+1
      }
      else if (substring(line,i,i) == 'R') {
        r_val[i] = r_val[i]+1
      }
      else if (substring(line,i,i) == 'S') {
        s_val[i] = s_val[i]+1
      }
      else if (substring(line,i,i) == 'T') {
        t_val[i] = t_val[i]+1
      }
      else if (substring(line,i,i) == 'V') {
        v_val[i] = v_val[i]+1 
      }
      else if (substring(line,i,i) == 'W') {
        w_val[i] = w_val[i]+1
      }
      else if (substring(line,i,i) == 'X') {
        x_val[i] = x_val[i]+1
      }
      else if (substring(line,i,i) == 'Y') {
        y_val[i] = y_val[i]+1
      }
      else if (substring(line,i,i) == 'Z') {
        z_val[i] = z_val[i]+1
      }
    }
  }
  
  LaPlace_pseudocount = 0.1
  equal_prob_nucleotide = 0.04
  
  for (i in 1:9){
    a_val[i] = round(log((a_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    b_val[i] = round(log((b_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    c_val[i] = round(log((c_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    d_val[i] = round(log((d_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    e_val[i] = round(log((e_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    f_val[i] = round(log((f_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    g_val[i] = round(log((g_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    h_val[i] = round(log((h_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    i_val[i] = round(log((i_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    j_val[i] = round(log((j_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    k_val[i] = round(log((k_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    l_val[i] = round(log((l_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    m_val[i] = round(log((m_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    n_val[i] = round(log((n_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    p_val[i] = round(log((p_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    q_val[i] = round(log((q_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    r_val[i] = round(log((r_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    s_val[i] = round(log((s_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    t_val[i] = round(log((t_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    v_val[i] = round(log((v_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    w_val[i] = round(log((w_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    x_val[i] = round(log((x_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    y_val[i] = round(log((y_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
    z_val[i] = round(log((z_val[i] + LaPlace_pseudocount)/(count_lines + 0.4)/equal_prob_nucleotide,2),3)
  }
  return (list(a_val=a_val, b_val=b_val, c_val=c_val, d_val=d_val, e_val=e_val,
               f_val=f_val, g_val=g_val, h_val=h_val, i_val=i_val, j_val=j_val,
               k_val=k_val, l_val=l_val, m_val=m_val, n_val=n_val, p_val=p_val,
               q_val=q_val, r_val=r_val, s_val=s_val, t_val=t_val, v_val=v_val,
               w_val=w_val, x_val=x_val, y_val=y_val, z_val=z_val))
}

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

cgr_pwm = function(data, org_seq,
                   seq.base = row.names(table(data)),
                   sf = F,
                   res = 100) {
  
  kmer = 9
  kmer_list = get_kmers(org_seq, kmer)
  pwm_vecs = pwm(data)
  
  ##### normalize the pwm vectors in [0,1] bounds #####
  a_min = min(pwm_vecs$a_val)
  for (i in 1:9){
    pwm_vecs$a_val[i] = pwm_vecs$a_val[i]+ (abs(a_min) + 1)
  }

  b_min = min(pwm_vecs$b_val)
  for (i in 1:9){
    pwm_vecs$b_val[i] = pwm_vecs$b_val[i]+ (abs(b_min) + 1)
  }
  
  c_min = min(pwm_vecs$c_val)
  for (i in 1:9){
    pwm_vecs$c_val[i] = pwm_vecs$c_val[i]+ (abs(c_min) + 1)
  }
  
  d_min = min(pwm_vecs$d_val)
  for (i in 1:9){
    pwm_vecs$d_val[i] = pwm_vecs$d_val[i]+ (abs(d_min) + 1)
  }
  
  e_min = min(pwm_vecs$e_val)
  for (i in 1:9){
    pwm_vecs$e_val[i] = pwm_vecs$e_val[i]+ (abs(e_min) + 1)
  }
  
  f_min = min(pwm_vecs$f_val)
  for (i in 1:9){
    pwm_vecs$f_val[i] = pwm_vecs$f_val[i]+ (abs(f_min) + 1)
  }
  
  g_min = min(pwm_vecs$g_val)
  for (i in 1:9){
    pwm_vecs$g_val[i] = pwm_vecs$g_val[i]+ (abs(g_min) + 1)
  }
  
  h_min = min(pwm_vecs$h_val)
  for (i in 1:9){
    pwm_vecs$h_val[i] = pwm_vecs$h_val[i]+ (abs(h_min)+1)
  }
  
  i_min = min(pwm_vecs$i_val)
  for (i in 1:9){
    pwm_vecs$i_val[i] = pwm_vecs$i_val[i]+ (abs(i_min)+1)
  }
  
  j_min = min(pwm_vecs$j_val)
  for (i in 1:9){
    pwm_vecs$j_val[i] = pwm_vecs$j_val[i]+ (abs(j_min)+1)
  }
  
  k_min = min(pwm_vecs$k_val)
  for (i in 1:9){
    pwm_vecs$k_val[i] = pwm_vecs$k_val[i]+ (abs(k_min)+1)
  }

  l_min = min(pwm_vecs$l_val)
  for (i in 1:9){
    pwm_vecs$l_val[i] = pwm_vecs$l_val[i]+ (abs(l_min)+1)
  }
  
  m_min = min(pwm_vecs$m_val)
  for (i in 1:9){
    pwm_vecs$m_val[i] = pwm_vecs$m_val[i]+ (abs(m_min)+1)
  }
  
  n_min = min(pwm_vecs$n_val)
  for (i in 1:9){
    pwm_vecs$n_val[i] = pwm_vecs$n_val[i]+ (abs(n_min)+1)
  }

  p_min = min(pwm_vecs$p_val)
  for (i in 1:9){
    pwm_vecs$p_val[i] = pwm_vecs$p_val[i]+ (abs(p_min)+1)
  }
  
  q_min = min(pwm_vecs$q_val)
  for (i in 1:9){
    pwm_vecs$q_val[i] = pwm_vecs$q_val[i]+ (abs(q_min)+1)
  }

  r_min = min(pwm_vecs$r_val)
  for (i in 1:9){
    pwm_vecs$r_val[i] = pwm_vecs$r_val[i]+ (abs(r_min)+1)
  }
  
  s_min = min(pwm_vecs$s_val)
  for (i in 1:9){
    pwm_vecs$s_val[i] = pwm_vecs$s_val[i]+ (abs(s_min)+1)
  }
  
  t_min = min(pwm_vecs$t_val)
  for (i in 1:9){
    pwm_vecs$t_val[i] = pwm_vecs$t_val[i]+ (abs(t_min)+1)
  }

  v_min = min(pwm_vecs$v_val)
  for (i in 1:9){
    pwm_vecs$v_val[i] = pwm_vecs$v_val[i]+ (abs(v_min)+1)
  }
  
  w_min = min(pwm_vecs$w_val)
  for (i in 1:9){
    pwm_vecs$w_val[i] = pwm_vecs$w_val[i]+ (abs(w_min)+1)
  }
  
  x_min = min(pwm_vecs$x_val)
  for (i in 1:9){
    pwm_vecs$x_val[i] = pwm_vecs$x_val[i]+ (abs(x_min)+1)
  }
  
  y_min = min(pwm_vecs$y_val)
  for (i in 1:9){
    pwm_vecs$y_val[i] = pwm_vecs$y_val[i]+ (abs(y_min)+1)
  }
  
  z_min = min(pwm_vecs$z_val)
  for (i in 1:9){
    pwm_vecs$z_val[i] = pwm_vecs$z_val[i]+ (abs(z_min)+1)
  }
  
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
  
  #check for input errors
  stopifnot(
    length(seq.base) >= length(table(data)),
    all(row.names(table(data)) %in% seq.base),
    sf <= 1,
    sf >= 0,
    res >= 1)
  
  #get the number of bases
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
  
  for (i in 1:length(kmer_list)){
    char_tmp = data
    k = 0
    for (j in i:(i+8)){
      k = k+1
      pt = pt + (unlist(base[char_tmp[j],]) - pt) * sf
      x[j] = pt[1]
      y[j] = pt[2]
      x.matrix = ceiling((x[j]+r ) * res/(2*r))
      y.matrix = ceiling((y[j]+r ) * res/(2*r))
      if(char_tmp[j] == "A"){
        add_val = pwm_vecs$a_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "B"){
        add_val = pwm_vecs$b_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "C"){
        add_val = pwm_vecs$c_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "D"){
        add_val = pwm_vecs$d_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "E"){
        add_val = pwm_vecs$e_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "F"){
        add_val = pwm_vecs$f_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "G"){
        add_val = pwm_vecs$g_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "H"){
        add_val = pwm_vecs$h_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "I"){
        add_val = pwm_vecs$i_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "J"){
        add_val = pwm_vecs$j_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "K"){
        add_val = pwm_vecs$k_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "L"){
        add_val = pwm_vecs$l_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "M"){
        add_val = pwm_vecs$m_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "N"){
        add_val = pwm_vecs$n_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "P"){
        add_val = pwm_vecs$p_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "Q"){
        add_val = pwm_vecs$q_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "R"){
        add_val = pwm_vecs$r_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "S"){
        add_val = pwm_vecs$s_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "T"){
        add_val = pwm_vecs$t_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "V"){
        add_val = pwm_vecs$v_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "W"){
        add_val = pwm_vecs$w_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "X"){
        add_val = pwm_vecs$x_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "Y"){
        add_val = pwm_vecs$y_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      else if(char_tmp[j] == "Z"){
        add_val = pwm_vecs$z_val[k]
        A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + add_val
      }
      
    }
    k=0
  }
  
  # }
  
  #return matrix, coordinates, scaling factor, resolution
  return(list(matrix = A,
              x = x,
              y = y,
              scaling_factor = sf,
              resolution = res,
              base = base))
}

mat <- np$load("sequenceFiles.npy", allow_pickle=TRUE)

for (i in 1:length(mat)) {
  tmp <- (mat[[i]])
  test = as.character(strsplit(tmp, "")[[1]])
  test_cgr = cgr_pwm(test, tmp,  res = 1)
  png(paste(i,".png"))
  plot(x=test_cgr$x, y=test_cgr$y, xlab='', ylab='', xaxt="n", yaxt="n")
  dev.off()
}

