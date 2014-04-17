open System
open System.IO

#r @"..\packages\MathNet.Numerics.2.6.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\MathNet.Numerics.FSharp.2.6.0\lib\net40\MathNet.Numerics.FSharp.dll"

open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Statistics

//Listings 8.1 - Standard regression function and data-importing functions
let loadDataSet (filename : string) = 
    let data = 
        File.ReadAllLines(filename)
        |> Array.map (fun line -> line.Split('\t'))
        |> Array.map (fun line -> 
               ([ if line.Length = 2 then yield line.[0] |> float
                  else 
                      for i in 0..line.Length - 2 do
                          yield line.[i] |> float ], (line.[line.Length - 1] |> float)))
    
    let dataMat = 
        [ for item in data -> fst item ]
    
    let labelMat = 
        [ for item in data -> snd item ]
    
    (dataMat, labelMat)

let datapath = __SOURCE_DIRECTORY__ + @"\ex0.txt"
let xArr, yArr = loadDataSet (datapath)

/// Compute the best fit line for linear regression
let standRegres xArr yArr = 
    let xMat = matrix xArr
    let yMat = (matrix (yArr |> fun item -> [ item ])).Transpose()
    let xTx = xMat.Transpose() * xMat
    if xTx.Determinant() = 0. then failwith "This matrix is singulat, cannot do inverse"
    else 
        let ws = xTx.Inverse() * (xMat.Transpose() * yMat)
        ws

let ws = standRegres xArr yArr

// 3.00774 ; 1.69532
// Locally weighted linear regression function
let lwlr (testPoint : Vector<float>) xArr yArr k = 
    let xMat = matrix xArr
    let yMat = (matrix (yArr |> fun item -> [ item ])).Transpose()
    let m = xMat.RowCount
    let weight = DenseMatrix.Identity(m)
    for j in 0..m - 1 do
        let diffMat = (testPoint - xMat.Row(j)).ToRowMatrix()
        weight.[j, j] <- Math.Exp((diffMat * diffMat.Transpose()).[0, 0] / (-2.0 * k ** 2.0))
    let xTx = xMat.Transpose() * (weight * xMat)
    if xTx.Determinant() = 0. then failwith "This matrix is singulat, cannot do inverse"
    else 
        let ws = xTx.Inverse() * (xMat.Transpose() * (weight * yMat))
        testPoint * ws

let lwlrTest (testArr : Matrix<float>) xArr yArr k = 
    let m = testArr.RowCount    
    let yHat = 
        vector [ for i in 0..m -> 0. ]
    for i in 0..m - 1 do
        yHat.[i] <- (lwlr (testArr.Row(i)) xArr yArr k).[0]
    yHat

let xArrMatrix = matrix xArr
let yArrVector = vector yArr

yArrVector.[0]

let testVector1 = lwlr (xArrMatrix.Row(0)) xArr yArr 1.0
// seq [3.122044714]
let testVector2 = lwlr (xArrMatrix.Row(0)) xArr yArr 0.001
// seq [3.201757286]
let yHat = lwlrTest xArrMatrix xArr yArr 0.003

//Example : predicting the age of an abalone
let datapathAbalone = __SOURCE_DIRECTORY__ + @"\abalone.txt"
let abX, abY = loadDataSet (datapathAbalone)
let abXMatrix = matrix abX
let abYVector = vector abY

abYVector.Count

let getSlice list startIndex endIndex = 
    list
    |> Seq.skip startIndex
    |> Seq.take endIndex
    |> Seq.toList

// working with 1x99 and 99x1 
let yHat01 = lwlrTest (matrix (getSlice abX 0 98)) (getSlice abX 0 98) (getSlice abY 0 98) 0.1
let yHat1 = lwlrTest (matrix (getSlice abX 0 98)) (getSlice abX 0 98) (getSlice abY 0 98) 1.
let yHat10 = lwlrTest (matrix (getSlice abX 0 98)) (getSlice abX 0 98) (getSlice abY 0 98) 10.

/// Discribe error of our estimate
let rssError (yArr : Matrix<float>) (yHatArr : Matrix<float>) = 
    let tmpMatrix = (yArr - yHatArr)
    tmpMatrix.MapInplace(fun item -> item ** 2.0)
    tmpMatrix.Column(0).Sum()

let err = rssError (abYVector.[0..98].ToColumnMatrix()) (yHat01.ToRowMatrix().Transpose())
let err2 = rssError (abYVector.[0..98].ToColumnMatrix()) (yHat1.ToRowMatrix().Transpose())
let err3 = rssError (abYVector.[0..98].ToColumnMatrix()) (yHat10.ToRowMatrix().Transpose())

//Ridge regression
let ridgeRegres (xMat : Matrix<float>) (yMat : Matrix<float>) lam =
    let xTx = xMat.Transpose() * xMat
    let m = xMat.ColumnCount
    let denom = xTx + DenseMatrix.Identity(m) * lam
    if denom.Determinant() = 0. then failwith "This matrix is singulat, cannot do inverse"
    else 
        let ws = denom.Inverse() * (xMat.Transpose() * yMat)
        ws

let ridgeTest xArr yArr =
    let xMat = matrix xArr
    let yMat = (matrix (yArr |> fun item -> [ item ])).Transpose()
    let yMean = yMat.ToRowWiseArray().Mean()
    yMat.MapInplace(fun item -> (item - yMean))
    let xMean = xMat.ToRowWiseArray().Mean()
    let xVar = new DescriptiveStatistics(xMat.ToRowWiseArray())
    xMat.MapInplace(fun item -> (item - xMean) / xVar.Variance)
    let numTest = 30
    let wMat = matrix [ for i in 0..numTest-1 -> [ for j in 0..xMat.ColumnCount-1 -> 0.]]
    for i in 0..numTest-1 do
        let ws = ridgeRegres xMat yMat (Math.Exp(float (i-10)))
        let wsArray = ws.Transpose().ToRowWiseArray()
        wMat.SetRow(i, wsArray) |> ignore
    wMat

let abX2,abY2 = loadDataSet datapathAbalone
let ridgeWeights = ridgeTest abX2 abY2

