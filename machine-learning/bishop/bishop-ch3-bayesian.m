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

function r = param_pdf(w0, w1, mu, sigma)
  assert(size_equal(w0, w1) && issquare(sigma) )

  w = [ w0'(:) w1'(:) ];  # Make it a n x 2 matrix to fit mvnpdf arguments
  r = mvnpdf(w, mu', sigma);
  r = reshape(r, size(w0))'; # reshape column order first
endfunction

function r = prior_pdf(w0, w1)
  global m_0;
  global S_0;

  r = param_pdf(w0, w1, m_0, S_0)
endfunction

function r = posterior_pdf(w0, w1, t, x)
  global m_0;
  global S_0;
  global beta_param;
  assert(size_equal(w0, w1) && isvector(t));

  Phi = [ ones(length(x), 1) x];
  S_n = (S_0^-1 + beta_param*Phi'*Phi)^-1;
  m_n = S_n*(S_0^-1*m_0 + beta_param*Phi'*t);

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

function plot_likelihood_pdf(t, x)
  global w_th;
  range = -1:.001:1;
  [w0, w1] = meshgrid(range, range);
  p = likelihood_pdf(t, x, w0, w1);
  colormap(jet)
  hold on;
  contourf(w0, w1, p, 200, 'LineStyle', 'None')
  scatter(w_th(1), w_th(2), 50, 'w', 'o', 'filled')
  hold off;
endfunction

function plot_param_pdf(m_n, S_n)
  global w_th;
  range = -1:.05:1;
  [w0, w1] = meshgrid(range, range);
  p = param_pdf(w0, w1, m_n, S_n);
  colormap(jet)
          hold on;
  contourf(w0, w1, p, 200, 'LineStyle', 'None')
          scatter(w_th(1), w_th(2), 50, 'w', 'o', 'filled')
          hold off;
endfunction

function plot_pdf(pdf)
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

function plot_prior_pdf()
  global w_th;
  range = -1:.05:1;
  [w0, w1] = meshgrid(range, range);
  p = prior_pdf(w0, w1);
  colormap(jet)
  hold on;
  contourf(w0, w1, p, 200, 'LineStyle', 'None')
  scatter(w_th(1), w_th(2), 50, 'w', 'o', 'filled')
  hold off;
endfunction

function plot_posterior_pdf(t, x)
  global w_th;
  range = -1:.05:1;
  [w0, w1] = meshgrid(range, range);
  p = posterior_pdf(w0, w1, t, x);
  colormap(jet)
  hold on;
  contourf(w0, w1, p, 200, 'LineStyle', 'None')
  scatter(w_th(1), w_th(2), 50, 'w', 'o', 'filled')
  hold off;
endfunction

clf;
subplot(4, 3, 1)
title('likelihood')
subplot(4, 3, 2)
title('prior/posterior')
plot_prior_pdf()
subplot(4, 3, 4)
plot_likelihood_pdf([0], [1])
subplot(4, 3, 5)
plot_posterior_pdf([0], [1])
subplot(4, 3, 7)
plot_likelihood_pdf([-.75], [-.6])
subplot(4, 3, 8)
plot_posterior_pdf([0; -.75], [1; -.6])

