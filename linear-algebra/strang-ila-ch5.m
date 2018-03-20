# Chapter 5.1 - ex 32
# For rand(n)
disp('rand()');
for n = [50 100 200 400]
  m = 20;
  dets = zeros(m,1);
  for i = 1:m
    dets(i) = abs(det(rand(n)));
  endfor
  n
  mu = mean(dets)
  v = std(dets)
endfor

# For randn(n)
disp('randn()');
for n = [50 100 200 400]
  m = 20;
  dets = zeros(m,1);
  for i = 1:m
    dets(i) = abs(det(randn(n)));
  endfor
  n
  mu = mean(dets)
  v = std(dets)
endfor

# Ex 5.1.33
function ret = maxDetSignMatrix(n, i, A=[], ret=[ -Inf []] )
  if (i > 0)
    ret = maxDetSignMatrix(n, i-1, [A  1], ret);
    ret = maxDetSignMatrix(n, i-1, [A -1], ret);
  else
    A_cand = reshape(A, n, n);
    d_cand = det(A_cand);
    if (d_cand > ret(1))
      ret = [ d_cand A ];
    endif
  endif
endfunction

# Chapter 5.2. - ex.3
A = [ 1 1 0; 1 0 1; 0 1 1]
C = [A zeros(3) ; zeros(3) A]  # Gives 4 and not -1

# Chapter 5.3 - ex.9
A = [ 1 1 -1 ; 1 -1 -1; -1 1 -1]
det(A) # Gives 4 Hadamard Matrix

# 5.3 - ex. 12
C = [ 3 2 1 ; 2 4 2; 1 2 3 ]  # Cofactor matrix of A
A = [ 2 -1 0 ; -1 2 -1 ; 0 -1 2 ]
C'
det(A)*A^-1
