function contour_f(f, range)
  if nargin() == 1
    range = -1:.05:1;
  endif
  [x, y] = meshgrid(range, range);
  z = f(x, y);
  colormap(jet)
  hold on;
  contourf(x, y, z, 200, 'LineStyle', 'None')
  hold off;
endfunction
