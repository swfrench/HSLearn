-- | Arbitrary order autoregressive model fitting
module AutoRegression 
( dataMatrix
, fit
, predict
) where

import Numeric.LinearAlgebra

type DVec = Vector Double
type DMat = Matrix Double

-- | Build data matrix for linear @p@-th order AR model
dataMatrix 
  :: Int  -- ^ Order @p@ of AR model
  -> DVec -- ^ Vector of data
  -> DMat -- ^ Returns: Data matrix
dataMatrix p xs = 
  let n = dim xs `div` p
      b = asColumn $ constant 1.0 n
      d = fromRows . takesV (replicate n p) $ xs
  in  fromBlocks [[b, d]]

predict 
  :: Int    -- ^ Order @p@ of the AR model
  -> DVec   -- ^ Vector of data
  -> DVec   -- ^ AR coefficients
  -> Int    -- ^ Data index @ix@ to predict
  -> Double -- ^ Returns: One-step prediction for index @ix@
predict p xs ys ix = 
  if    ix < p
  then  error $ "Cannot predict at index " ++ show ix ++ " for p = " ++ show p
  else  ys @> 0 + subVector 1 p ys <.> subVector (ix - p) p xs

-- | Fit (forward-prediction, least-squares) an @p@-th order AR model to the given data vector
fit 
  :: Int  -- ^ Order of AR model
  -> DVec -- ^ Vector of data
  -> DVec -- ^ Returns: AR coefficients
fit p xs = 
  let n    = dim xs `div` p
      xmat = dataMatrix p xs
      ys   = subVector p (n - p + 1) xs
  in  xmat <\> ys
