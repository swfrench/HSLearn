
-- | Non-linear conjugate gradient algorithm
module ConjugateGradient 
( SearchConfig (..)
, defaultSearchConfig
, conjugateGradient
) where

import Numeric.LinearAlgebra

-- | Configuration for the backtracking line-search
data SearchConfig = SearchConfig 
  { tau   :: Double -- ^ geometric step-reduction factor
  , c1    :: Double -- ^ Nocedal and Wright (Wolfe conditions)
  , c2    :: Double -- ^ Nocedal and Wright (Wolfe conditions)
  , a0    :: Double -- ^ initial step size
  , niter :: Int    -- ^ number of search iterations
  } deriving (Show)

-- | Convenient default search configuration for non-linear CG
defaultSearchConfig :: SearchConfig
defaultSearchConfig = SearchConfig { tau = 0.5, c1 = 1.0e-4, c2 = 0.1, a0 = 1.0, niter = 100 }

-- Inexact back-tracking line search (Wolfe conditions)
search
  :: (Vector Double -> Double)
  -> (Vector Double -> Vector Double)
  -> Vector Double
  -> Vector Double
  -> SearchConfig
  -> Double
search cost grad p x conf = 
  let j0 = cost x
  in  wolfe 0 (a0 conf) j0 (a0 conf) j0 (p <.> grad x)
  where
    wolfe n an j0 amin jmin pg =  
      -- check number of iters; return best step if exhausted
      if    n == niter conf
      then  amin
      else
        let xn    = x + scale an p  -- seach point
            jn    = cost xn         -- search point cost
            pgn   = p <.> grad xn   -- search direction projected into gradient at search point
        -- test Wolfe conditions
        in  if    jn  <= j0 + c1 conf * an * pg && 
                  pgn >= c2 conf * pg
            then  an
            else
              let aminn = if jn < jmin then an else amin
                  jminn = if jn < jmin then jn else jmin
              in  wolfe (n + 1) (an * tau conf) j0 aminn jminn pg

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
