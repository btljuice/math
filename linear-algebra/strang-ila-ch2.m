# Chapter 2 - 1.30
u = [ 1 ; 0];
A = [.8 .3 ; .2 .7];
x = u;
k= [0 : 7];

while size(x,2) <= 7
   u = A*u;
   x = [ x u ];
end
plot(k,x);

# Chapter 2 - 1.34
A = [  2 -1  0  0 ;
      -1  2 -1  0 ;
       0 -1  2 -1 ;
       0  0  -1 2 ];
b = [ 1 2 3 4 ]'
