
-- | Gradient descent
module GradientDescent 
( gradientDescent
) where

import Numeric.LinearAlgebra

-- | Gradient Descent
gradientDescent 
  :: Double                            -- ^ learning rate
  -> Double                            -- ^ residual tolerance
  -> Integer                           -- ^ maximum number of iterations
  -> (Vector Double -> Double)         -- ^ cost function
  -> (Vector Double -> Vector Double)  -- ^ gradient function
  -> Vector Double                     -- ^ initial parameter estimate
  -> (Integer, Double, Vector Double)  -- ^ returns: (niter, res, theta)
gradientDescent alpha tol niter cost grad = optimize 0
  where optimize n t = let r = cost t in 
          if n == niter || r < tol 
            then (n, r, t)
            else optimize (n + 1) (t - scale alpha (grad t))
