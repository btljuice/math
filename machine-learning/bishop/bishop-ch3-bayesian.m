pkg load 'statistics'
pkg load 'geometry'

# Following equations of p.154 in chapter 3
global alpha = 2;  # prior precision
global beta_param = 25;  # Noise precision
global w_th = [ -.3; .5 ];
global m_0 = [ 0; 0 ];  # Prior mean
global S_0 = alpha^-1*eye(2); # Prior Covariance Matrix

function r = f_th(x)
  global w_th;

  x0 = ones(length(x), 1);
  r = [ x0 x ]*w_th;
endfunction

function r = sample_param_pdf(mu, sigma, n)
  r = mvnrnd(mu', sigma, n);
endfunction

function r = sample_prior_pdf(n)
  global m_0;
  global S_0;

  r = sample_param_pdf(m_0, S_0, n);
endfunction

function r = sample_posterior_pdf(n, t, x)
  [m_n, S_n] = posterior_mean_and_variance(t, x);

  r = sample_param_pdf(m_n, S_n, n);
endfunction

function r = param_pdf(w0, w1, mu, sigma)
  assert(size_equal(w0, w1) && issquare(sigma) );

  w = [ w0'(:) w1'(:) ];  # Make it a n x 2 matrix to fit mvnpdf arguments
  r = mvnpdf(w, mu', sigma);
  r = reshape(r, size(w0))'; # reshape column order first
endfunction

function r = prior_pdf(w0, w1)
  global m_0;
  global S_0;

  r = param_pdf(w0, w1, m_0, S_0)
endfunction

function [m_n, S_n] = posterior_mean_and_variance(t, x)
  global m_0;
  global S_0;
  global beta_param;
  assert(isvector(t) && size(x,1) == length(t));

  Phi = [ ones(length(x), 1) x];
  S_n = (S_0^-1 + beta_param*Phi'*Phi)^-1;
  m_n = S_n*(S_0^-1*m_0 + beta_param*Phi'*t);
endfunction

function r = posterior_pdf(w0, w1, t, x)
  assert(size_equal(w0, w1) && isvector(t));

  [m_n, S_n] = posterior_mean_and_variance(t, x);
  r = param_pdf(w0, w1, m_n, S_n);
endfunction

# t, vector : 1d target values
# x, vector : 1d predictor values
function ll = likelihood_pdf(t, x, w0, w1)
  global beta_param;
  noise_stdev = beta_param^-.5;
  assert(size_equal(x,t) && size_equal(w0, w1) && length(x) >= 1);

  ll = ones(size(w0));
  for i = 1:length(x)
    t_pred = w0 + w1*x(i);
    ## ll = ll .* stdnormal_pdf((t(i) - t_pred)/noise_stdev)/noise_stdev;
    ll = ll .* normpdf(t(i), t_pred, noise_stdev);
  end
endfunction

function contour_pdf(pdf)
  global w_th;
  range = -1:.05:1;
  [w0, w1] = meshgrid(range, range);
  p = pdf(w0, w1);
  colormap(jet)
  hold on;
  contourf(w0, w1, p, 200, 'LineStyle', 'None')
  scatter(w_th(1), w_th(2), 50, 'w', 'o', 'filled')
  hold off;
endfunction

function plot_2dlines(w, pts_t, pts_x)
  global w_th;

  n = size(w, 1);
  x = -1:.05:1;
  hold on;
  for i = 1:size(w, 1)
    plot(x, w(i, 1) + w(i, 2).*x, '-r')
  end
  plot(x, w_th(1) + w_th(2)*x, '-g')
  if nargin == 3
    scatter(pts_x, pts_t, 'b', 'o')
  endif
  xlim([-1, 1])
  ylim([-1, 1])
  hold off;
endfunction

x = rand([20 1]);
t = f_th(x) + normrnd(0, beta_param^-1, [20 1] );
x = [1; -.6; x ];
t = [0; -.75; t ];
clf;
subplot(4, 3, 1)
title('likelihood')
subplot(4, 3, 2)
title('prior/posterior')
contour_pdf(@prior_pdf)
subplot(4, 3, 3)
title('data space')
plot_2dlines(sample_prior_pdf(5))
subplot(4, 3, 4)
contour_pdf(@(w0, w1) likelihood_pdf(t(1), x(1), w0, w1))
subplot(4, 3, 5)
contour_pdf(@(w0, w1) posterior_pdf(w0, w1, t(1), x(1)))
subplot(4, 3, 6)
plot_2dlines(sample_posterior_pdf(5, t(1), x(1)), t(1), x(1))
subplot(4, 3, 7)
contour_pdf(@(w0, w1) likelihood_pdf(t(2), x(2), w0, w1))
subplot(4, 3, 8)
contour_pdf(@(w0, w1) posterior_pdf(w0, w1, t(1:2), x(1:2)))
subplot(4, 3, 9)
plot_2dlines(sample_posterior_pdf(5, t(1:2), x(1:2)), t(1:2), x(1:2))
subplot(4, 3, 10)
contour_pdf(@(w0, w1) likelihood_pdf(t, x, w0, w1))
subplot(4, 3, 11)
contour_pdf(@(w0, w1) posterior_pdf(w0, w1, t, x))
subplot(4, 3, 12)
plot_2dlines(sample_posterior_pdf(5, t, x), t, x)
