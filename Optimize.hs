
-- | Common optimization strategies
module Optimize
( gradientDescentFixed
, gradientDescentSearch
, conjugateGradient
) where

import Numeric.LinearAlgebra
import LineSearch

-- | Gradient descent with fixed learning rate
gradientDescentFixed
  :: Double                            -- ^ learning rate
  -> Double                            -- ^ residual tolerance
  -> Int                               -- ^ maximum number of iterations
  -> (Vector Double -> Double)         -- ^ cost function
  -> (Vector Double -> Vector Double)  -- ^ gradient function
  -> Vector Double                     -- ^ initial parameter estimate
  -> (Int, Double, Vector Double)      -- ^ returns: (niter, res, theta)
gradientDescentFixed alpha tol ngditer cost grad = optimize 0
  where optimize n t = let r = cost t in
          if n == ngditer || r < tol
          then  (n, r, t)
          else  optimize (n + 1) (t - scale alpha (grad t))

-- | Gradient descent with backtracking line search
gradientDescentSearch
  :: SearchConfig                      -- ^ configuration for backtracking line search
  -> Double                            -- ^ residual tolerance
  -> Int                               -- ^ maximum number of iterations
  -> (Vector Double -> Double)         -- ^ cost function
  -> (Vector Double -> Vector Double)  -- ^ gradient function
  -> Vector Double                     -- ^ initial parameter estimate
  -> (Int, Double, Vector Double)      -- ^ returns: (niter, res, theta)
gradientDescentSearch conf tol ngditer cost grad = optimize 0
  where optimize n t = let r = cost t in
          if n == ngditer || r < tol
          then  (n, r, t)
          else
            let alpha = search conf cost grad (- grad t) t
            in  optimize (n + 1) (t - scale alpha (grad t))

-- | Non-linear (Fletcher-Reeves) conjugate gradient algorithm (no restarts)
conjugateGradient
  :: SearchConfig                       -- ^ configuration for backtracking line search
  -> Double                             -- ^ residual tolerance
  -> Int                                -- ^ maximum number of iterations
  -> (Vector Double -> Double)          -- ^ cost function
  -> (Vector Double -> Vector Double)   -- ^ gradient function
  -> Vector Double                      -- ^ initial parameter estimate
  -> (Int, Double, Vector Double)       -- ^ returns: (niter, res, theta)
conjugateGradient conf tol ncgiter cost grad t0 =
  -- initialization steps
  let p = - grad t0
      a = search conf cost grad p t0
  in  cg 1 (t0 + scale a p) p p
  -- later steps
  where cg n t s p = let r = cost t in
          if    n >= ncgiter || r < tol
          then  (n, r, t)
          else
            let pn  = - grad t
                bn  = (pn <.> pn) / (p <.> p) -- Fletcher-Reeves
                sn  = pn + scale bn s
                an  = search conf cost grad sn t
            in  cg (n + 1) (t + scale an sn) sn pn
