
module Main where

import Text.Printf
import System.Environment (getArgs)
import System.Exit

import Control.DeepSeq
import Data.Time

import qualified Data.Vector.Unboxed as U

import KMeans (kmeansInitPP, euclidean) -- , manhattan)

type DVec  = U.Vector Double

loadData 
  :: FilePath 
  -> IO [DVec]
loadData f = parse =<< readFile f
  where parse = return . map (U.fromList . map read . words) . lines

saveData 
  :: FilePath 
  -> [DVec] 
  -> IO ()
saveData f = writeFile f . build
  where build = unlines . map (unwords . map show . U.toList)

parseArgs
  :: [String]
  -> IO (Int,Int,FilePath)
parseArgs (sk:sn:fn:[]) = return ( read sk :: Int, read sn :: Int, fn :: FilePath)
parseArgs _             = printf "Usage: prog k niter in.data\n" >> exitSuccess

withTiming :: IO a -> IO (Double,a)
withTiming f = do
  t0  <- getCurrentTime
  res <- f
  t1  <- getCurrentTime
  return (realToFrac $ diffUTCTime t1 t0, res)

mdsq :: NFData a => a -> IO ()
mdsq x = x `deepseq` return ()

main :: IO ()
main = do 
  (k,niter,fname) <- getArgs >>= parseArgs
  xs <- loadData fname
  mdsq xs
  printf "data is ready\n"
  printf "running kmeans ... "
  (dt, clusters) <- withTiming $ do
    cs <- kmeansInitPP k niter euclidean xs
    mdsq cs
    return cs
  printf "done\n"
  printf "computation took: %.4f s\n" dt
  saveData "out.test" (fst clusters)
  printf "saved result to out.test\n"

