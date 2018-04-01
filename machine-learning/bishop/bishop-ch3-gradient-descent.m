pkg load 'statistics'
pkg load 'geometry'

rand("seed", 42);  # Remove if needed

global w_th = [ -.3; .5 ];
global noise_stdev = .25;

global n_samples = 25;

function r = h_theorical(X)
  global w_th;

  r = X*w_th;
endfunction

function r = generate_noise(n)
  global noise_stdev;
  r = normrnd(0, noise_stdev, [ n 1 ] );
endfunction

function X = generate_predictors(n)
  X = [ ones([n 1]) rand([n 1]) ];
endfunction

function [X, T] = generate_samples(n)
  X = generate_predictors(n);
  T = h_theorical(X) + generate_noise(n);
endfunction

function l = loss(T_pred, T)
  # Sum of square errors computed for each column separately
  l = sum((T_pred - T).^2);
endfunction

function T = model_predict(W, X)
  # Linear regression model
  T = X*W;
endfunction

function l = loss_contour(w0, w1, X, T)
  W = [ w0(:)' ; w1(:)' ];
  T_pred = model_predict(W, X);
  n = size(T_pred, 2);
  l = loss(T_pred, repmat(T, [1 n]));
  l = reshape(l, size(w0));
endfunction

function W = batch_gradient_descent()
endfunction

# Sample x and t
[X , T] = generate_samples(n_samples);

contour_f(@(w0, w1) loss_contour(w0, w1, X, T), -1:.1:1);
contour3_f(@(w0, w1) loss_contour(w0, w1, X, T), -1:.1:1, -2:.2:2);
