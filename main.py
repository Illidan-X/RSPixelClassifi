import os
import re
from datetime import datetime as dt
import numpy as np
from numpy.random import shuffle
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def rastersample(raster, shpdir):
    idx = 0
    X_list = []
    Y_list = []
    filelist = os.listdir(shpdir)
    for filename in filelist:
        if re.split('\.', filename)[-1] == 'shp':
            filepath = shpdir+'/'+filename
            gdf = gpd.read_file(filepath)
            coord_list = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
            X_list += [x for x in raster.sample(coord_list)]
            Y_list += [idx] * len(coord_list)
            idx += 1

    X_train = np.array(X_list)
    Y_train = np.array(Y_list)

    idx = np.arange(len(X_list))
    shuffle(idx)
    X_train = X_train[idx]
    Y_train = Y_train[idx]

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=Y_train, s=50)
    # plt.show()

    return X_train, Y_train


def rasterCLF(raster, X_train, Y_train):
    bands = [np.ravel(raster.read(band)) for band in range(1, 4)]

    X_test = np.vstack((bands[0], bands[1], bands[2]))
    X_test = X_test.T

    model = {'LR': make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', multi_class='multinomial')),
             'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
             'NB': GaussianNB(),
             'SVM': make_pipeline(StandardScaler(), NuSVC(nu=0.21, kernel='rbf')),
             'RF': RandomForestClassifier(n_estimators=57, random_state=0)}

    model['LR'].fit(X_train, Y_train)

    Y_pred = model['LR'].predict(X_test)

    return Y_pred


def write2raster(raster, arr, path):
    gridcount = raster.width * raster.height
    band1 = np.empty(gridcount)
    band2 = np.empty(gridcount)
    band3 = np.empty(gridcount)

    color_list = [[235, 51, 36], [119, 67, 66], [117, 249, 77], [50, 130, 246], [240, 134, 80], [115, 43, 245]]
    for item in range(len(color_list)):
        band1[arr == item] = color_list[item][0]
        band2[arr == item] = color_list[item][1]
        band3[arr == item] = color_list[item][2]

    with rasterio.open(path, mode='w', driver='GTiff', width=raster.width, height=raster.height, count=3,
                       crs=raster.crs, transform=raster.transform, dtype='uint8') as dst:
        dst.write(band1.reshape(raster.width, raster.height), 1)
        dst.write(band2.reshape(raster.width, raster.height), 2)
        dst.write(band3.reshape(raster.width, raster.height), 3)


def main():
    rasterpath = './Raster/Landsat8OLI.tif'
    shpdir ='./SHP'

    src = rasterio.open(rasterpath)

    X_train, Y_train = rastersample(src, shpdir)
    pred = rasterCLF(src, X_train, Y_train)

    # 设置分类结果影像的文件名
    newpath = './Raster/LR.tif'

    write2raster(src, pred, newpath)

    src.close()


if __name__ == '__main__':
    dt1 = dt.now()
    main()
    print(f'Duration: {dt.now() - dt1}')
