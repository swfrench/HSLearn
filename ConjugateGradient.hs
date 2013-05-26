
-- | Non-linear conjugate gradient algorithm
module ConjugateGradient 
( conjugateGradient
) where

import Numeric.LinearAlgebra
import LineSearch

-- | Non-linear (Fletcher-Reeves) conjugate gradient algorithm (no restarts)
conjugateGradient 
  :: Double                             -- ^ residual tolerance
  -> Int                                -- ^ maximum number of iterations
  -> (Vector Double -> Double)          -- ^ cost function
  -> (Vector Double -> Vector Double)   -- ^ gradient function
  -> Vector Double                      -- ^ initial parameter estimate
  -> SearchConfig                       -- ^ configuration for backtracking line search
  -> (Int, Double, Vector Double)       -- ^ returns: (niter, res, theta)
conjugateGradient tol ncgiter cost grad t0 conf = 
  -- initialization steps
  let p = - grad t0 
      a = search cost grad p t0 conf
  in  cg 1 (t0 + scale a p) p p
  -- later steps
  where cg n t s p = let r = cost t in
          if    n >= ncgiter || r < tol 
          then  (n, r, t)
          else 
            let pn  = - grad t
                bn  = (pn <.> pn) / (p <.> p) -- Fletcher-Reeves
                sn  = pn + scale bn s
                an  = search cost grad sn t conf
            in  cg (n + 1) (t + scale an sn) sn pn
