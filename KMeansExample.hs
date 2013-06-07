-- Example k-means clustering
module Main where

import Text.Printf
import Control.DeepSeq
import Data.Time

import qualified Data.Vector.Unboxed as U

import KMeans (kmeansInitPP, euclidean) -- , manhattan)

-- Convenient type alias
type DVec  = U.Vector Double

-- Load the test data
loadVectors
  :: FilePath
  -> IO [DVec]
loadVectors f = parse =<< readFile f
  where parse = return . map (U.fromList . map read . words) . lines

-- Save resulting centroids
saveVectors
  :: FilePath
  -> [DVec]
  -> IO ()
saveVectors f = writeFile f . build
  where build = unlines . map (unwords . map show . U.toList)

-- Run some action in the IO monad w/ timing
-- N.B. it is up to the action @f@ to ensure its result is fully evaluated
withTiming :: IO a -> IO (Double,a)
withTiming f = do
  t0  <- getCurrentTime
  res <- f
  t1  <- getCurrentTime
  return (realToFrac $ diffUTCTime t1 t0, res)

-- Wrapper for comon @deepseq@ use case
mdsq :: NFData a => a -> IO ()
mdsq x = x `deepseq` return ()

-- Example clustering
main :: IO ()
main = do
  -- config
  let k     = 3
      niter = 100
      fname = "in.test"
  -- load features
  xs <- loadVectors fname
  mdsq xs
  printf "data is ready\n"
  -- run clustering w/ timing
  printf "running kmeans ... "
  (dt, clusters) <- withTiming $ do
    cs <- kmeansInitPP k niter euclidean xs
    mdsq cs
    return cs
  printf "done\n"
  -- report timing ans save resulting centroids
  printf "computation took: %.4f s\n" dt
  saveVectors "out.test" (fst clusters)
  printf "saved result to out.test\n"
