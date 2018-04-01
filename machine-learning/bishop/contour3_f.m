function contour3_f(f, xrange, yrange)
  if nargin() == 1
    xrange = yrange = -1:.05:1;
  elseif nargin() == 2
    yrange = xrange;
  endif
  [x, y] = meshgrid(xrange, yrange);
  z = f(x, y);
  ## contour3(x, y, z, 30)
  ## meshc(x, y, z)
  surfc(x, y, z)
endfunction
