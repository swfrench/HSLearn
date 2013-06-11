{-# LANGUAGE BangPatterns #-}

-- | k-means clustering algorithm with multiple initialization schemes
module KMeans
( euclidean
, manhattan
, label
, distortion
, kmeans
, kmeansInitRandom
, kmeansInitPP
) where

import Control.Monad (liftM)
import System.Random (randomRIO)
import qualified Data.Vector.Unboxed as U

-- Type alias for a vector of unboxed @Double@s
type DVec  = U.Vector Double

-- Type alias for the distance measures
type Distance = DVec -> DVec -> Double

-- Strict calculation of the mean of a list of data points ('DVec's)
mean
  :: [DVec]
  ->  DVec
mean    []  = error "Cannot take the mean of an empty list"
mean (x:xs) = final . foldl stepl (1 :: Int,x) $ xs
  where stepl (!n,!s) z = (n+1,U.zipWith (+) s z)
        final (!n,!s)   = U.map (/ fromIntegral n) s

-- | Squared Euclidean distance between two data vectors
euclidean :: Distance
euclidean !x !y = U.sum $ U.zipWith f x y
  where f !u !v = (u - v) ** 2

-- | Manhattan distance between two data vectors
manhattan :: Distance
manhattan !x !y = U.sum $ U.zipWith f x y
  where f !u !v = abs (u - v)

-- | Apply labels to a set of data vectors given a set of cluster centroids
label
  :: Distance -- ^ Distance measure
  -> [DVec]   -- ^ Cluster centroids
  -> [DVec]   -- ^ Data vectors
  -> [Int]    -- ^ Returns: data labels (cluster ids)
label dist cs = map go
  where go x = snd . minimum $ zip (map (dist x) cs) [0..]

-- Calculate updated centroid estimates for each cluster
update
  ::  Int    -- Number of clusters @k@
  -> [DVec]  -- Data @xs@
  -> [Int]   -- Data labels @ls@ given current centroids
  -> [DVec]  -- Returns: new estimate of cluster centroids
update k xs ls = map centroid [0..k-1]
  where centroid n = mean . map snd . filter ((==n) . fst) $ zip ls xs

-- | Summed misfit for cluster configuration
distortion
  :: Distance -- ^ Distance measure
  -> [DVec]   -- ^ Cluster centroids
  -> [Int]    -- ^ Data labels given current centroids
  -> [DVec]   -- ^ Data
  -> Double   -- ^ Returns: summed misfit over all data vectors
distortion dist cs ls xs =
  foldl (\a (l,x) -> a + dist x (cs !! l)) 0 $ zip ls xs

-- Random centroid initialization
initRandom
  :: Int       -- Number of centroids requested @k@
  -> [DVec]    -- Data @xs@
  -> IO [DVec] -- Returns: initial centroids from @xs@
initRandom k xs = go k ([],[])
  where n = length xs
        go m s =
          if    m == 0
          then  return . fst $ s
          else  randomRIO (0,n-1) >>= next m s
        next m s@(cs,ixs) ix =
          if    ix `elem` ixs || (xs !! ix `elem` cs)
          then  go m s
          else  go (m-1) (xs !! ix:cs, ix:ixs)

-- Centroid initialization for the k-means++ technique
initPP
  :: Int        -- Number of centroids requested @k@
  -> [DVec]     -- Data @xs@
  -> Distance   -- Distance measure
  -> IO [DVec]  -- Returns: initial centroids from @xs@
initPP k xs dst = go k ([],[])
  where n = length xs
        go 0 s = return . fst $ s
        go m s =
          if    m == k
          then  randomRIO (0,n-1) >>= next m s
          else  selectPP xs s dst >>= next m s
        next m (cs,ixs) ix = go (m-1) (xs !! ix:cs, ix:ixs)

-- Inner workings of k-means++ centroid selection
selectPP
  :: [DVec]         -- Data @xs@
  -> ([DVec],[Int]) -- Current centroids @cs@ and indices in @xs@
  -> Distance       -- Distance measure
  -> IO Int         -- Returns: index of newly selected centroid from @xs@
selectPP xs (cs,ixs) dst = inner
  where mindst x = minimum . map (dst x) $ cs
        cdf      = init . scanl (+) 0 . map mindst $ xs
        sample x = length (takeWhile (<=x) cdf) - 1
        select x = let ix = sample x in
          if    ix `elem` ixs
          then  inner      -- no luck, try again
          else  return ix  -- this is a new index, return it
        inner = randomRIO (0, last cdf) >>= select

-- | The k-means clustering algorithm.
-- The user must supply list of @k@ initial centroids (type @['DVec']@)
kmeans
  :: Int            -- ^ Number of iterations @n@
  -> Distance       -- ^ Distance measure @dist@; 'euclidean', 'manhattan', or arbitrary user-defined
  -> [DVec]         -- ^ Data vectors @xs@
  -> [DVec]         -- ^ Initial cluster centroids @cs@
  -> ([DVec],[Int]) -- ^ Returns: tuple of solution cluster centroids @cs'@ and list of integer labels @ls@
kmeans n dist xs cs = go n cs
  where go m cs' = let ls = label dist cs' xs in
          if    m == 0
          then  (cs',ls)
          else  go (m-1) (update k xs ls)
        k = length cs

-- | The k-means clustering algorithm, with initial cluster centroids randomly
-- chosen from the data.
kmeansInitRandom
  :: Int                -- ^ Number of clusters @k@
  -> Int                -- ^ Number of iterations @n@
  -> Distance           -- ^ Distance measure @dist@; 'euclidean', 'manhattan', or arbitrary user-defined
  -> [DVec]             -- ^ Data vectors @xs@
  -> IO ([DVec],[Int])  -- ^ Returns: tuple of solution cluster centroids @cs'@ and list of integer labels @ls@ in the IO Monad
kmeansInitRandom k n dist xs =
  kmeans n dist xs `liftM` initRandom k xs

-- | The k-means clustering algorithm, with initial cluster centroids chosen
-- from the data using the k-means++ of Arthur & Vassilvitskii (2007).
kmeansInitPP
  :: Int                -- ^ Number of clusters @k@
  -> Int                -- ^ Number of iterations @n@
  -> Distance           -- ^ Distance measure @dist@; 'euclidean', 'manhattan', or arbitrary user-defined
  -> [DVec]             -- ^ Data vectors @xs@
  -> IO ([DVec],[Int])  -- ^ Returns: tuple of solution cluster centroids @cs'@ and list of integer labels @ls@ in the IO Monad
kmeansInitPP k n dist xs =
  kmeans n dist xs `liftM` initPP k xs dist
