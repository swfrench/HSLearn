-- Example one-vs-all logistic regression classifier for the MNIST digits dataset
module Main where

import Data.Time
import Numeric.LinearAlgebra
import Control.DeepSeq
import Text.Printf (printf)

import MNIST
import LogisticRegression
import LineSearch (defaultSearchConfig)
import Optimize (conjugateGradient, CGBnType(FletcherReeves))

-- Convenient type aliases
type Vec = Vector Double
type Mat = Matrix Double

-- Build the feature matrix w/ leading bias term
withBias :: Int -> [Vec] -> Mat
withBias m images = fromBlocks [[asColumn $ constant 1.0 m, fromRows images]]

-- Train the one-vs-all logistic regression model for the specified label
train :: Int -> [Vec] -> [Int] -> Int -> (Int, Double, Vec)
train m images labels label =
  let xs = withBias m images
      ys = fromList $ map (\l -> if l == label then 1.0 else 0.0) labels
      fc = cost xs ys
      fg = gradient xs ys
      -- lr = 0.01
      -- fc = costRegularized lr xs ys
      -- fg = gradientRegularized lr xs ys
      t0 = constant 0.0 (cols xs)
  in  conjugateGradient FletcherReeves defaultSearchConfig 0.0001 32 fc fg t0

-- Train one-vs-all models
buildClassifiers :: Int -> Int -> [Vec] -> [Int] -> [(Int,Double,Vec)]
buildClassifiers m n images labels = map (train m images labels) [0..n-1]

-- Index of a maximum list element
argmax :: Ord a => [a] -> Int
argmax xs = snd . maximum $ zip xs [0..]

-- Using the set of trained one-vs-all models, classify each of the supplied examples
classify :: Int -> [Vec] -> [Vec] -> [Int]
classify m images thetas =
  let xs = withBias m images
      hs = toLists . fromColumns . map (hypothesis xs) $ thetas
  in  map argmax hs

-- Calculate the success rate of the supplied classification
success :: Int -> [Int] -> [Int] -> Double
success m labels predictions =
  let wrong = sum $ zipWith (\l p -> if l == p then 0.0 else 1.0) labels predictions
  in  100 * (1.0 - wrong / fromIntegral m)

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

-- Example regression
main :: IO ()
main = do
  -- load data
  (dtLoad,(vm,vLabels,vImages,tm,tLabels,tImages)) <- withTiming $ do
    (MNISTData vm _ _ vLabels vImages) <- loadMNISTData $ validationData { dataPath = "mnist_data" }
    (MNISTData tm _ _ tLabels tImages) <- loadMNISTData $ trainingData   { dataPath = "mnist_data" }
    mdsq vLabels
    mdsq vImages
    mdsq tLabels
    mdsq tImages
    return (vm,vLabels,vImages,tm,tLabels,tImages)
  printf "read in data: %i training, %i validation (%fs elapsed)\n" tm vm dtLoad
  -- train
  (dtTrain,thetas) <- withTiming $ do
    let (_,_,thetas) = unzip3 $ buildClassifiers tm 10 tImages tLabels
    mdsq thetas
    return thetas
  printf "trained classifier (%fs elapsed)\n" dtTrain
  -- trainign performance
  (dtLabelTrain, pLabelsTrain) <- withTiming $ do
    let pLabelsTrain = classify tm tImages thetas
    mdsq pLabelsTrain
    return pLabelsTrain
  printf "success rate on training: %f (%fs elapsed)\n" (success tm tLabels pLabelsTrain) dtLabelTrain
  -- validation performance
  (dtLabelValidation, pLabelsValidation) <- withTiming $ do
    let pLabelsValidation = classify vm vImages thetas
    mdsq pLabelsValidation
    return pLabelsValidation
  printf "success rate on validation: %f (%fs elapsed)\n" (success vm vLabels pLabelsValidation) dtLabelValidation
