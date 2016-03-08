from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import numpy as np
import shapefile
import sys
import random
import csv
import pandas as pd
import matplotlib as mpl
import os

def mapping(states, targetCounties, checkPoints, clientName):

    shapePathRoot = 'C:\\Users\\dwright\\code\\geo_shapes\\'
    countyShapeName = 'cb_2014_us_county_500k'


    shapePathCounty = manageShapeFile(shapePathRoot, countyShapeName)
    print shapePathCounty

    #testShape = shapefile.Reader(shapePathCounty)

    stateShapeName = 'cb_2014_us_state_500k'
    shapePathState = manageShapeFile(shapePathRoot, stateShapeName)
    stateShape = shapefile.Reader(shapePathState)

    #tp = testShape.shapeType
    #if tp not in [0, 1, 3, 5, 8]:
    #    shapepath = convertShapefile(shapePathCounty)

    f = open('C:\\Users\\dwright\\code\\state_codes.csv','rb')
    rdr = csv.reader(f)
    rdr.next()
    stateLookup = {i[2].zfill(2):i[1] for i in rdr}
    codeLookup = {stateLookup[i]:i for i in stateLookup.keys()}

    stateCodes = [codeLookup[i] for i in states]
    minLat, minLon, maxLat, maxLon = getBoundaries(shapePathCounty, targetCounties, stateCodes)

    fig = plt.figure()
    # axes: left, bottom, width, height

    # building two subplots, one for the map and one for the colorbar
    # this function sets how many 'rows and columns' are used
    # so subplot2grid(100,1)uses 100 rows and 1 column
    # the
    ax1 = plt.subplot2grid((100, 1), (0, 0), rowspan=80)
    #ax2 = plt.subplot2grid((100, 1), (89, 0))
    #ax2.get_yaxis().set_visible(False)
    #ax2.get_xaxis().set_visible(False)

    #ax2 = fig.add_subplot(211)
    #ax2 = fig.add_axes([.05, .05, .8, .2])
    #ax1 = fig.add_axes([.05, .3, .9, .6])

    print 'coordinate boundaries', minLat, minLon, maxLat, maxLon
    centerLat = (minLat + maxLat) / 2
    centerLon = (minLon + maxLon) / 2

    map = Basemap(llcrnrlon=minLon,llcrnrlat=minLat,urcrnrlon=maxLon,urcrnrlat=maxLat,
                  projection='aeqd', lat_0=centerLat, lon_0=centerLon, resolution='l')
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    #map.fillcontinents(color=(.8,.8,.8), lake_color='aqua')
    #map.fillcontinents(lake_color='aqua')
    #map.drawstates(linewidth=4)
    map.shadedrelief()
    #map.drawcounties(linewidth=.1)


    map.readshapefile(shapePathCounty, 'counties', drawbounds=True)

    # info are the fields
    for info, shape in zip(map.counties_info, map.counties):
        #print info
        #print shape
        #print 'name', info['NAME']
        #sys.exit()
        patches = []
        #print info['NAME'] + '_' + stateLookup[info['STATEFP']]
        if stateLookup[info['STATEFP']] + '_' + info['NAME'] in targetCounties.keys():
            col = targetCounties[stateLookup[info['STATEFP']] + '_' + info['NAME']][0]
            origValue = targetCounties[stateLookup[info['STATEFP']] + '_' + info['NAME']][1]
            patches.append(Polygon(np.array(shape), True))
            ax1.add_collection(PatchCollection(patches, facecolor=col, edgecolor='black',
                                              linewidths=1., zorder=2, alpha=0.5))

    cols = [checkPoints['bottom'][1], checkPoints['mid'][1], checkPoints['top'][1]]
    print cols

    cmap = mpl.colors.ListedColormap(cols)
    cmap.set_over('1')
    cmap.set_under('0')

    # length of boundary array needs to be one more than length of color list
    # and needs to be monotonically increasing

    #bounds = [checkPoints['bottom'][0] -1,checkPoints['bottom'][0], checkPoints['mid'][0], checkPoints['top'][0]]
    #print bounds
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm#, boundaries=bounds
    #                               , extend='both',
                                # Make the length of each extension
                                # the same as the length of the
                                # interior colors:
    #                            extendfrac='auto', ticks=bounds, spacing='uniform', orientation='horizontal')

    #cb.set_label('Custom extension lengths, some other units')
    # lon, lat =
    #xpt, ypt = map(lon, lat)
    #lonpt, latpt = map(xpt, ypt, inverse=True)
    # I can also find the center of the county and name it!
    # will need to figure out eh size of the name

    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color='aqua')
    # draw lat/lon grid lines every 30 degrees.

    title = 'Exposure Heatmap for {0}'.format(clientName)
    plt.title(title)
    outFileName = 'ExposureMap_{0}'.format(clientName)
    outPath = 'c:\\users\\dwright\\dropbox\\{0}.jpg'.format(outFileName)

    plt.tight_layout()
    plt.savefig(outPath)
    #plt.show()
    plt.clf()

    return

def manageShapeFile(shapePathRoot, shapeName):

    shapeFileList = os.listdir(shapePathRoot)
    convertedName = shapeName + '_converted' + '.shp'
    print shapeName
    if shapeName + '.shp' not in shapeFileList:
        print 'no shape file', shapeName + '.shp'
        sys.exit()
    # test if we need to convert
    elif convertedName not in shapeFileList:
        shapePath = convertShapefile(shapePathRoot + shapeName)
    # no need to convert
    else:
        shapePath = shapePathRoot + convertedName

    shapePath = shapePath.replace('.shp', '')
    return shapePath


def getShapeFile(state, county, path):

    outPath = path + '_working_state'
    # is it not possible for me to pass a shapefile around without
    # writing it to disk? come on!

    return outPath

def getBoundaries(read_path, counties, stateCodes):
    print 'getting boundaries for:', stateCodes
    sf = shapefile.Reader(read_path)
    tp = sf.shapeType
    shapeRecs = sf.records()
    shapeShapes = sf.shapes()
    shapeFields = sf.fields
    shapeIndices = range(len(shapeRecs))
    #print sf.fields
    minLon = 0
    maxLon = 0
    maxLat = 0
    minLat = 0
    for i in shapeIndices:
        shpe = shapeShapes[i].points
        rec = shapeRecs[i]
        shpeAr = np.array(shpe)
        #print rec

        if rec[0] in stateCodes:
            #print shpeAr

            thisLon, thisLat = np.amin(shpeAr, 0)
            if minLon == 0 or minLon > thisLon:
                minLon = thisLon
            if minLat == 0 or minLat > thisLat:
                minLat = thisLat

            thisLon, thisLat = np.amax(shpeAr, 0)
            if maxLon == 0 or maxLon < thisLon:
                maxLon = thisLon
            if maxLat == 0 or maxLat < thisLat:
                maxLat = thisLat

    return minLat - 1, minLon - 1, maxLat+ 1, maxLon+ 1

def convertShapefile(read_path):
    print 'converting shapefile'
    sf = shapefile.Reader(read_path)
    tp = sf.shapeType
    print tp

    shapeRecs = sf.records()
    shapeShapes = sf.shapes()
    shapeFields = sf.fields
    print sf.fields


    #Value Shape Type
    #0 Null Shape
    #1 Point
    #3 PolyLine
    #5 Polygon
    #8 MultiPoint
    #11 PointZ -- these are 3d versions of the above
    #13 PolyLineZ
    #15 PolygonZ
    #18 MultiPointZ
    #21 PointM -- not sure what these are versions of...
    #23 PolyLineM
    #25 PolygonM
    #28 MultiPointM
    #31 MultiPatch

    to_2D_dict = {0: 0, 11: 1, 13: 3, 15: 5, 18: 8}
    print tp, to_2D_dict[tp]
    newtp = to_2D_dict[tp]
    wrtr = shapefile.Writer(shapeType=newtp)
    write_path = read_path + '_converted'
    print len(shapeRecs), len(shapeShapes)

    shapeIndices = range(len(shapeRecs))
    # loop through the shapefiles and create a 2D version of the shapefile we want
    for f in shapeFields:
        #print f
        #wrtr.field(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8])
        # for each type of field, we need to pass them individually, not as a list.. ugh
        wrtr.field(f[0], f[1], f[2], f[3])
    for i in shapeIndices:
        if i % 10 == 0:
            print i - len(shapeRecs)
        shpe = shapeShapes[i].points
        rec = shapeRecs[i]
        #print 'points:', shpe
        #print 'new type:', newtp

        #print 'record:', rec
        # next line gives error, iteration over non-sequence
        wrtr.poly(shapeType=newtp, parts=[shpe])
        wrtr.record(rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7], rec[8])

    wrtr.save(write_path)

    return write_path

def getData(fname, bottomPercentile, midPoint, topPercentile, p):

    baseData = pd.DataFrame.from_csv(p + fname, index_col=False)
    origValues = baseData.ix[:, 0]
    #print baseData
    #print baseData.ix[:,0]
    midPoint = 0.5
    bottom = baseData.ix[:, 0].astype('float').quantile(bottomPercentile)
    top = baseData.ix[:, 0].astype('float').quantile(topPercentile)
    dataMid = baseData.ix[:, 0].astype('float').quantile(midPoint)

    dataRange = top - bottom

    # cap the data at the top
    baseData.ix[baseData.ix[:,0] > top, 0] = top
    # subtract the bottom
    baseData.ix[:,0] = baseData.ix[:,0] - bottom
    # minimize at 0
    baseData.ix[baseData.ix[:,0] < 0, 0] = 0
    baseData.ix[baseData.ix[:,0] > 1, 0] = baseData.ix[:,0] / (dataRange)

    # these are the three colors we are using as the range
    low = [255, 51, 51]
    med = [255, 255, 102]
    high = [102, 255, 102]
    low = [i/float(255) for i in low]
    med = [i/float(255) for i in med]
    high = [i/float(255) for i in high]

    locs = baseData.shape[0]
    r = pd.DataFrame({'r': baseData.ix[:, 0]})
    g = pd.DataFrame({'g': baseData.ix[:, 0]})
    b = pd.DataFrame({'b': baseData.ix[:, 0]})

    #here I am working on doing this with matrices
    lowAr = np.tile(np.array(low), (locs,1))
    medAr = np.tile(np.array(med), (locs,1))
    highAr = np.tile(np.array(high), (locs,1))
    targetCounties = dict()
    for i in range(locs):
        test = baseData.ix[i, 0]
        # here we check against midPoint, which is 0.5 by default
        if test < midPoint:
            rs = ((med[0] - low[0]) * test / midPoint + low[0])
            gs = ((med[1] - low[1]) * test / midPoint + low[1])
            bs = ((med[2] - low[2]) * test / midPoint + low[2])
        else:
            rs = ((high[0] - med[0]) * (test - midPoint) / (1-midPoint) + med[0])
            gs = ((high[1] - med[1]) * (test - midPoint) / (1-midPoint) + med[1])
            bs = ((high[2] - med[2]) * (test - midPoint) / (1-midPoint) + med[2])
        r.ix[i, 0] = rs
        g.ix[i, 0] = gs
        b.ix[i, 0] = bs
        stateName = str(baseData.ix[i, 1]).upper()
        countyName = str(baseData.ix[i, 2]).replace(' COUNTY', '').title()
        targetCounties[stateName + '_' + countyName] = ((rs, gs, bs), origValues.ix[i, 0])

    states = baseData.ix[:, 1].unique()
    #add = pd.concat([r,g,b], axis=1)
    #outData = pd.concat([baseData, add], axis=1)

    checkPoints = {'bottom': (bottom, low), 'mid': (dataMid, med), 'top': (top, high)}
    return states, targetCounties, checkPoints

def main():

    bottomPercentile = 0
    midPoint = 0.5
    topPercentile = 1
    p = 'c:\\users\\dwright\\dropbox\\'
    #p = '\\\\BA-FS-NY\Data\\Yr 2016\\Orchid\\Spinnaker Project\\Cat Modeling\\'


    #for clientName in ['Orchid','Reference']:
    for clientName in ['Loudoun', 'GUA']:

        fname = '{0}Exposure.csv'.format(clientName)

        print 'getting data for', clientName
        targetStates, targetCounties, checkPoints = getData(fname, bottomPercentile, midPoint, topPercentile, p)
        print 'mapping', clientName
        mapping(targetStates, targetCounties, checkPoints, clientName)

if __name__ == '__main__':
    main()