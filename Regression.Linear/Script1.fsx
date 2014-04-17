open System
open System.IO

#r @"..\packages\MathNet.Numerics.2.6.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\MathNet.Numerics.FSharp.2.6.0\lib\net40\MathNet.Numerics.FSharp.dll"

open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double

//Listings 8.1 - Standard regression function and data-importing functions

let loadDataSet (filename : string) =
    let data =
        File.ReadAllLines(filename)
        |> Array.map (fun line -> line.Split('\t'))
        |> Array.map (fun line -> 
                ([ (line.[0] |> float) ; (line.[1] |> float) ], (line.[2] |> float)))
    let dataMat = [ for item in data -> fst item]
    let labelMat = [ for item in data -> snd item ]
    (dataMat, labelMat)

let datapath = __SOURCE_DIRECTORY__ + @"\ex0.txt"

let xArr,yArr = loadDataSet(datapath)

/// Compute the best fit line for linear regression
let standRegres xArr yArr =
    let xMat = matrix xArr
    let yMat = (matrix (yArr |> fun item -> [item])).Transpose()
    let xTx  = xMat.Transpose() * xMat
    if xTx.Determinant() = 0. then
        failwith "This matrix is singulat, cannot do inverse"
    else
        let ws = xTx.Inverse() * (xMat.Transpose() * yMat)
        ws

let ws = standRegres xArr yArr
// 3.00774 ; 1.69532

// Locally weighted linear regression function
let lwlr (testPoint : Vector<float>) xArr yArr k  =
    let xMat = matrix xArr
    let yMat = (matrix (yArr |> fun item  -> [item])).Transpose()
    let m = xMat.RowCount
    let weight = DenseMatrix.Identity(m)
    
    for j in 0..m-1 do
        let diffMat = (testPoint - xMat.Row(j)).ToRowMatrix()
        weight.[j,j] <- Math.Exp((diffMat * diffMat.Transpose()).[0,0] / (-2.0 * k ** 2.0))
    
    let xTx = xMat.Transpose() * (weight * xMat)
    if xTx.Determinant() = 0. then 
        failwith "This matrix is singulat, cannot do inverse"
    else
        let ws = xTx.Inverse() * (xMat.Transpose() * (weight * yMat))
        testPoint * ws 

let lwlrTest (testArr : Matrix<float>) xArr yArr k =
    let m = testArr.RowCount
    let yHat = vector [ for i in 0..m -> 0.]
    for i in 0..m do
        yHat.[i] <- (lwlr (testArr.Row(i)) xArr yArr k).[0]
    yHat

let xArrMatrix  = matrix xArr
let yArrVector  = vector yArr
yArrVector.[0]
let testVector1 = lwlr (xArrMatrix.Row(0)) xArr yArr 1.0
// seq [3.122044714]
let testVector2 = lwlr (xArrMatrix.Row(0)) xArr yArr 0.001
// seq [3.201757286]

let yHat = lwlrTest xArrMatrix xArr yArr 0.003
yHat