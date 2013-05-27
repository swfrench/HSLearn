-- | Common optimization strategies
module Optimize
( LearnRate (..)
, CGBnType  (..)
, gradientDescent
, conjugateGradient
) where

import Numeric.LinearAlgebra
import LineSearch

type Vec = Vector Double

-- | Learning rate for gradient descent (fixed vs. search)
data LearnRate =
    FixedRate   Double       -- ^ Fixed learning rate (specified)
  | SearchRate  SearchConfig -- ^ Backtracking line search with config (specified)
  deriving (Show,Eq)

-- | Gradient descent with fixed learning rate
gradientDescent
  :: LearnRate          -- ^ configuration for learning rate
  -> Double             -- ^ residual tolerance
  -> Int                -- ^ maximum number of iterations
  -> (Vec -> Double)    -- ^ cost function
  -> (Vec -> Vec)       -- ^ gradient function
  -> Vec                -- ^ initial parameter estimate
  -> (Int, Double, Vec) -- ^ returns: (@niter@, @res@, @theta@)
gradientDescent (FixedRate alpha) tol ngditer cost grad = optimize 0
  where optimize n t = let r = cost t in
          if n == ngditer || r < tol
          then  (n, r, t)
          else  optimize (n + 1) (t - scale alpha (grad t))
gradientDescent (SearchRate conf) tol ngditer cost grad = optimize 0
  where optimize n t = let r = cost t in
          if n == ngditer || r < tol
          then  (n, r, t)
          else
            let alpha = search conf cost grad (- grad t) t
            in  optimize (n + 1) (t - scale alpha (grad t))

-- | Formula for the beta_n term in non-linear CG
data CGBnType =
    FletcherReeves  -- ^ Fletcher-Reeves formula
  | PolakRibiere    -- ^ Polak-Ribiere formula
  deriving (Show,Eq)

-- | Non-linear conjugate gradient algorithm (no explicit restarts)
conjugateGradient
  :: CGBnType           -- ^ configuration for beta_n term
  -> SearchConfig       -- ^ configuration for backtracking line search
  -> Double             -- ^ residual tolerance
  -> Int                -- ^ maximum number of iterations
  -> (Vec -> Double)    -- ^ cost function
  -> (Vec -> Vec)       -- ^ gradient function
  -> Vec                -- ^ initial parameter estimate
  -> (Int, Double, Vec) -- ^ returns: (@niter@, @res@, @theta@)
conjugateGradient FletcherReeves = conjugateGradientBase (\pn p -> (pn <.> pn)       / (p <.> p))
conjugateGradient PolakRibiere   = conjugateGradientBase (\pn p -> (pn <.> (pn - p)) / (p <.> p))
conjugateGradientBase fbn conf tol ncgiter cost grad t0 =
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
                bn  = max 0.0 (fbn pn p)
                sn  = pn + scale bn s
                an  = search conf cost grad sn t
            in  cg (n + 1) (t + scale an sn) sn pn
