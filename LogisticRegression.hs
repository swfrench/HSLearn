
-- | Logistic regression hypothesis function, as well as both normal and 
-- regularized variants of the cost and gradient
module LogisticRegression 
( hypothesis
, cost
, gradient
, costRegularized
, gradientRegularized
) where

import Numeric.LinearAlgebra
import Foreign.Storable (Storable)

ndim 
  :: (Storable a, Num b) 
  => Vector a 
  -> b 
ndim = fromIntegral . dim

sigmoid 
  :: Vector Double 
  -> Vector Double
sigmoid xs = 1.0 / (1.0 + mapVector exp (-xs))

hypothesis 
  :: Matrix Double 
  -> Vector Double 
  -> Vector Double
hypothesis xs theta = sigmoid $ xs <> theta

cost 
  :: Matrix Double  -- ^ examples
  -> Vector Double  -- ^ labels
  -> Vector Double  -- ^ parameters
  -> Double         -- ^ returns: cost (no regularization)
cost xs ys theta = 
  let m    = ndim ys
      hs   = hypothesis xs theta
      logs = mapVector log
  in - (ys <.> logs hs + (1.0 - ys) <.> logs (1.0 - hs)) / m
        
gradient 
  :: Matrix Double  -- ^ examples
  -> Vector Double  -- ^ labels
  -> Vector Double  -- ^ parameters
  -> Vector Double  -- ^ returns: gradient (no regularization)
gradient xs ys theta = 
  let m  = ndim ys
      hs = hypothesis xs theta
      rs = hs - ys
  in trans xs <> rs / m
 
costRegularized
  :: Double 
  -> Matrix Double 
  -> Vector Double 
  -> Vector Double 
  -> Double
costRegularized lambda xs ys theta = 
  let m  = ndim ys
      w  = 0.5 * lambda / m
      ts = subVector 1 (dim theta - 1) theta
  in cost xs ys theta + w * (ts <.> ts)
 
gradientRegularized
  :: Double 
  -> Matrix Double 
  -> Vector Double 
  -> Vector Double 
  -> Vector Double
gradientRegularized lambda xs ys theta = 
  let m  = ndim ys
      w  = lambda / m
      ts = join [0.0, subVector 1 (dim theta - 1) theta]
  in gradient xs ys theta + scale w ts
