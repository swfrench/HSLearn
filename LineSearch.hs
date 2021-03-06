-- | Backtracking line search (Wolfe conditions)
module LineSearch
( WolfeType (..)
, SearchConfig (..)
, defaultSearchConfig
, wolfe
, search
) where

import Numeric.LinearAlgebra

-- Convenient type alias
type Vec = Vector Double

-- | Wolfe conditions
data WolfeType = Wolfe | StrongWolfe deriving (Show,Eq)

-- | Configuration for the backtracking line-search
data SearchConfig = SearchConfig
  { tau   :: Double     -- ^ geometric step-reduction factor
  , a0    :: Double     -- ^ initial step size
  , niter :: Int        -- ^ number of search iterations
  , c1    :: Double     -- ^ constant for Armijo rule
  , c2    :: Double     -- ^ constant for curvature condition
  , wtype :: WolfeType  -- ^ type of Wolfe conditions enforced
  } deriving (Show,Eq)

-- | Convenient default search configuration
--
--  Appropriate for non-linear CG
defaultSearchConfig :: SearchConfig
defaultSearchConfig = SearchConfig
  { tau   = 0.5
  , a0    = 1.0
  , niter = 100
  , c1    = 1.0e-4
  , c2    = 1.0e-1
  , wtype = Wolfe }

-- | Check whether Wolfe conditions are satisfied
wolfe
  :: SearchConfig -- ^ contains config for Wolfe conditions
  -> Double       -- ^ initial cost
  -> Double       -- ^ current search-point cost
  -> Double       -- ^ current step size
  -> Double       -- ^ search direction projected into initial gradient
  -> Double       -- ^ search direction projected into gradient at search point
  -> Bool         -- ^ returns: whether Wolfe conditions satisfied
wolfe conf j0 jn an pg pgn =
  jn <= j0 + c1 conf * an * pg && -- Armijo rule
  case wtype conf of
    Wolfe       -> pgn     >= c2 conf * pg
    StrongWolfe -> abs pgn <= c2 conf * abs pg

-- | Inexact back-tracking line search (Wolfe conditions)
search
  :: SearchConfig     -- ^ configuration for search
  -> (Vec -> Double)  -- ^ cost function
  -> (Vec -> Vec)     -- ^ gradient function
  -> Vec              -- ^ search direction
  -> Vec              -- ^ current solution
  -> Double           -- ^ returns: best step size
search conf cost grad p x =
  let j0  = cost x
      pg  = p <.> grad x
      -- initial (quadratic) guess for step size?
      ep  = 1.0e-8                  -- small test perturbation
      bb  = 8.0                     -- overshoot
      jep = cost $ x + scale ep p   -- misfit at test perturbation
      aa0 = - bb * pg * ep ** 2 / (2 * (jep - j0 - ep * pg))
      -- or hard-wired default
      -- aa0 = a0 conf
  in  iter 0 aa0 j0 aa0 j0 pg
  where
    iter n an j0 amin jmin pg =
      -- check number of iters; return best step if niter exhausted
      if    n == niter conf
      then  amin
      else
        let xn  = x + scale an p  -- seach point
            jn  = cost xn         -- search point cost
            pgn = p <.> grad xn   -- search direction projected into gradient at search point
        -- test Wolfe conditions
        in  if    wolfe conf j0 jn an pg pgn
            then  an
            else
              let aminn = if jn < jmin then an else amin
                  jminn = if jn < jmin then jn else jmin
              in  iter (n + 1) (an * tau conf) j0 aminn jminn pg
