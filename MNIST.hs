-- | The MNIST hand-written digits dataset
--
-- Data files must be downloaded separately from <http://yann.lecun.com/exdb/mnist/>
module MNIST
( MNISTData (..)
, DataFiles (..)
, trainingData
, validationData
, loadMNISTData
) where

import Data.Binary.Strict.Get
import qualified Data.ByteString as B
import Control.Monad (replicateM)
import qualified Numeric.LinearAlgebra as LA

type Vec = LA.Vector Double

-- | Dataset representation: @MNISTData m rows cols labels images@
data MNISTData = MNISTData Int Int Int [Int] [Vec]

-- | Combines the MNIST label and image data file names, along with a path the directory in which they reside
data DataFiles = DataFiles { labelsFile :: FilePath, imagesFile :: FilePath, dataPath :: FilePath } deriving (Show,Eq)

-- | Convenient default for the training dataset
trainingData :: DataFiles
trainingData = DataFiles "train-labels-idx1-ubyte" "train-images-idx3-ubyte" "."

-- | Convenient default for the validation dataset
validationData :: DataFiles
validationData = DataFiles "t10k-labels-idx1-ubyte" "t10k-images-idx3-ubyte" "."

labelsMagic :: Int
labelsMagic = 2049

imagesMagic :: Int
imagesMagic = 2051

getLabels :: Get (Int,Int,[Int])
getLabels = do
  magic   <- fmap fromIntegral getWord32be
  m       <- fmap fromIntegral getWord32be
  labels  <- replicateM m (fmap fromIntegral getWord8)
  return (magic, m, labels)

getImages :: Get (Int,Int,Int,Int,[Vec])
getImages = do
  magic   <- fmap fromIntegral getWord32be
  m       <- fmap fromIntegral getWord32be
  rows    <- fmap fromIntegral getWord32be
  cols    <- fmap fromIntegral getWord32be
  images  <- replicateM m . fmap LA.fromList . replicateM (rows * cols) $ fmap fromIntegral getWord8
  return (magic, m, rows, cols, images)

-- | Loads the specified data (sub)set
--
-- Use 'trainingData' or 'validationData' as convenient defaults, possibly 
-- changing the 'dataPath' if necessary.
loadMNISTData :: DataFiles -> IO MNISTData
loadMNISTData dataFiles = do
  -- data sources
  labelsSource <- B.readFile $ dataPath dataFiles ++ "/" ++ labelsFile dataFiles
  imageSource  <- B.readFile $ dataPath dataFiles ++ "/" ++ imagesFile dataFiles
  let ((Right (lmagic, lm, labels), _))             = runGet getLabels labelsSource
  let ((Right (imagic, im, rows, cols, images), _)) = runGet getImages imageSource
  -- checks
  if lmagic /= labelsMagic
  then 
    error $ "loadMNISTData: bad magic value on " ++ labelsFile dataFiles
  else if imagic /= imagesMagic
  then 
    error $ "loadMNISTData: bad magic value on " ++ imagesFile dataFiles
  else if lm /= im
  then 
    error "loadMNISTData: example count mismatch between labels and data"
  else 
    return $ MNISTData lm rows cols labels images
