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

-- Convenient type aliases
type Vec = Vector Double
type Mat = Matrix Double

-- Tail of a Vec
tailVector
  :: Vec
  -> Vec
tailVector x = subVector 1 (dim x - 1) x

-- Dimension of a vector, recast to any @Num@
ndim
  :: (Storable a, Num b)
  => Vector a
  -> b
ndim = fromIntegral . dim

-- Logistic sigmoid
sigmoid
  :: Vec
  -> Vec
sigmoid xs = 1.0 / (1.0 + mapVector exp (-xs))

-- | Logistic regression hypothesis function
hypothesis
  :: Mat
  -> Vec
  -> Vec
hypothesis xs theta = sigmoid $ xs <> theta

-- | Logistic regression (unregularized) cost function
cost
  :: Mat    -- ^ examples
  -> Vec    -- ^ labels
  -> Vec    -- ^ parameters
  -> Double -- ^ returns: cost (no regularization)
cost xs ys theta =
  let m    = ndim ys
      hs   = hypothesis xs theta
      logs = mapVector log
  in - (ys <.> logs hs + (1.0 - ys) <.> logs (1.0 - hs)) / m

-- | Logistic regression (unregularized) cost gradient function
gradient
  :: Mat  -- ^ examples
  -> Vec  -- ^ labels
  -> Vec  -- ^ parameters
  -> Vec  -- ^ returns: gradient (no regularization)
gradient xs ys theta =
  let m  = ndim ys
      hs = hypothesis xs theta
      rs = hs - ys
  in trans xs <> rs / m

-- | Logistic regression (regularized) cost function
costRegularized
  :: Double -- ^ regularization factor
  -> Mat    -- ^ examples
  -> Vec    -- ^ labels
  -> Vec    -- ^ parameters
  -> Double -- ^ returns: cost (with regularization)
costRegularized lambda xs ys theta =
  let m  = ndim ys
      w  = 0.5 * lambda / m
      ts = tailVector theta
  in cost xs ys theta + w * (ts <.> ts)

-- | Logistic regression (regularized) cost gradient function
gradientRegularized
  :: Double -- ^ regularization factor
  -> Mat    -- ^ examples
  -> Vec    -- ^ labels
  -> Vec    -- ^ parameters
  -> Vec    -- ^ returns: gradient (with regularization)
gradientRegularized lambda xs ys theta =
  let m  = ndim ys
      w  = lambda / m
      ts = join [0.0, tailVector theta]
  in gradient xs ys theta + scale w ts
