import pandas as Panda
import numpy as np
import os
import glob

def k5FoldCrossValidation(k=1):
    tb = bacaXLSX()
    X,y = getXyValue(300, 1500, tb)
    actual = getActualValue(0, 300, tb)
    data = getDataToPredict(0, 300, tb)
    preds = Predict(data, X, y, k)
    BikinTabelOutput(preds, actual, "output1")
    
    X,y = getXyValue(600, 1500, tb)
    actual = getActualValue(300, 600, tb)
    data = getDataToPredict(300, 600, tb)
    preds = Predict(data, X, y, k)
    BikinTabelOutput(preds, actual, "output2",index=300)
    
    X,y = getXyValue(900, 1500, tb)
    actual = getActualValue(600, 900, tb)
    data = getDataToPredict(600, 900, tb)
    preds = Predict(data, X, y, k)
    BikinTabelOutput(preds, actual, "output3",index=600)
    
    X,y = getXyValue(0, 1000, tb)
    actual = getActualValue(900, 1200, tb)
    data = getDataToPredict(900, 1200, tb)
    preds = Predict(data, X, y, k)
    BikinTabelOutput(preds, actual, "output4",index=900)
    
    X,y = getXyValue(0, 1200, tb)
    actual = getActualValue(1200, 1500, tb)
    data = getDataToPredict(1200, 1500, tb)
    preds = Predict(data, X, y, k)
    BikinTabelOutput(preds, actual, "output5",index=1200)

def bacaXLSX():
    return Panda.read_excel(r"DataSetTB3_SHARE.xlsx",header=0)

def validasi(arrA, arrB):
    qty = len(arrA)
    correct = 0
    for i in range(qty):
        if arrA[i] == arrB[i]:
            correct+=1
    return str((correct/qty)*100)

def exportCSV(fileName, listOfTable):
    fileName = "./tmp/"+fileName
    np.savetxt(fileName+".csv", 
            listOfTable,
            delimiter =", ", 
            fmt ='% s')

def getXyValue(start,end,tb):
    X = tb.drop("idData",axis=1)
    X = tb.drop("label",axis=1)
    y = tb["label"]
    return X.values[start:end], y.values[start:end]

def BikinTabelOutput(Prediksi, Aktual, fileName, index=1):
    temp = []
    temp.append(["idData","Label Aktual","Hasil Klasifikasi","Akurasi"])
    temp.append([index,Aktual[0],Prediksi[0],validasi(Prediksi, Aktual)+"%"])
    index += 1
    for i in range(1,len(Prediksi)):
        temp.append([index,Aktual[i],Prediksi[i]])
        index+=1
    exportCSV(fileName, temp)

def getActualValue(start,end,tb):
    actual = tb["label"]
    return actual.values[start:end]

def Predict(data,X,y,k):
    preds = []
    for i in data:
        jarak = np.linalg.norm(X - i, axis=1)
        nnId = jarak.argsort()[:k]
        nnLabel = y[nnId]
        prediction = int(nnLabel.mean())
        preds.append(prediction)
    return preds

def combineToExcel():
    path = "./tmp"
    all_files = glob.glob(os.path.join(path, "*.csv"))
    writer = Panda.ExcelWriter('output.xlsx', engine='xlsxwriter')
    df_from_each_file = (Panda.read_csv(f) for f in all_files)
    for idx, df in enumerate(df_from_each_file):
        df.to_excel(writer, sheet_name='output{0}.csv'.format(idx))
    writer.save()    

def getDataToPredict(start,end,tb):
    datas = tb.drop("idData",axis=1)
    datas = tb.drop("label",axis=1)
    return datas.values[start:end]

if __name__ == "__main__":
    k5FoldCrossValidation(k=1) 
    combineToExcel()
