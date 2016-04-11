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
import matplotlib.cm as cm
import os
import itertools


def mapping(states, targetCounties, minVal, maxVal, clientName, outPath, shapePathRoot):

    # shapePathRoot = 'C:\\Python27\\Lib\\site-packages\\mpl_toolkits\\basemap\\data\\'

    countyShapeName = 'cb_2014_us_county_500k'
    # countyShapeName = 'UScounties'

    shapePathCounty = manageShapeFile(shapePathRoot, countyShapeName)
    # print shapePathCounty
    # testShape = shapefile.Reader(shapePathCounty)

    stateShapeName = 'cb_2014_us_state_500k'
    shapePathState = manageShapeFile(shapePathRoot, stateShapeName)
    stateShape = shapefile.Reader(shapePathState)

    f = open('state_codes.csv','rb')
    rdr = csv.reader(f)
    rdr.next()
    stateLookup = {i[2].zfill(2):i[1] for i in rdr}
    codeLookup = {stateLookup[i]:i for i in stateLookup.keys()}
    exStates = ['AK','HI']
    stateCodes = [codeLookup[i] for i in states if i not in exStates]
    exCodes = [i for i in states if i  in exStates]
    print 'getting boundaries and excluding', exCodes
    minLat, minLon, maxLat, maxLon = getBoundaries(shapePathCounty, targetCounties, stateCodes)

    fig = plt.figure()
    # axes: left, bottom, width, height

    ax1 = plt.subplot(111)

    print 'coordinate boundaries', minLat, minLon, maxLat, maxLon
    centerLat = (minLat + maxLat) / 2
    centerLon = (minLon + maxLon) / 2
    lats = [minLat, maxLat, centerLat]
    lons = [minLon, maxLon, centerLon]
    cords = list(itertools.chain.from_iterable([[(i,j) for i in lons] for j in lats]))

    convCords = []
    # convert the midpoint lat and min lon to coordiantes and make the x of that the lower corner

    map = Basemap(llcrnrlon=minLon-2,llcrnrlat=minLat-2,urcrnrlon=maxLon,urcrnrlat=maxLat,
                  projection='stere', resolution='l',lat_0=centerLat, lon_0=centerLon, lat_ts=minLat)
    # draw coastlines, country boundaries, fill continents.
    for cord in cords:
      convCords.append(map(cord[0],cord[1]))
    
    convCordsAr = np.array(convCords)
    #print convCordsAr
    mins = np.amin(convCordsAr,axis=1)
    maxes = np.amax(convCordsAr,axis=1)
    xmin = mins[0]
    xmax = maxes[0]
    ymin = mins[1]
    ymax = maxes[1]

    map = Basemap(llcrnrlon=minLon-2,llcrnrlat=minLat-2,urcrnrlon=maxLon,urcrnrlat=maxLat,
                  projection='stere', resolution='l',lat_0=centerLat, lon_0=centerLon, lat_ts=minLat,
                  width=xmax-xmin, height=ymax-ymin)

    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    #map.fillcontinents(color=(.8,.8,.8), lake_color='aqua')
    #map.fillcontinents(lake_color='aqua')
    #map.drawstates(linewidth=4)
    map.shadedrelief()
    #map.drawcounties(linewidth=.1)


    map.readshapefile(shapePathCounty, 'counties', drawbounds=True)
    norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal, clip=False)
    cmap = cm.get_cmap(name='Reds')
    colMaker = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # info are the fields
    myCols = []
    for info, shape in zip(map.counties_info, map.counties):
        patches = []

        if stateLookup[info['STATEFP']] + '_' + info['NAME'] in targetCounties.keys():

            origValue = targetCounties[stateLookup[info['STATEFP']] + '_' + info['NAME']]

            col = colMaker.to_rgba(origValue)
            myCols.append(origValue)
            patches.append(Polygon(np.array(shape), closed=True, label=stateLookup[info['STATEFP']]))

            ptch = PatchCollection(patches, edgecolor='black', facecolor=col,
                                       linewidths=1., alpha=0.7)
            coll = ax1.add_collection(ptch)
    coll.set_array(np.array(myCols))

    #ptch.set_array(np.array(origValue))


    cbar = map.colorbar(coll, location='bottom', pad='5%', cmap='Reds')
    cbar.set_label('Location Value')
    #map.readshapefile(shapePathState, 'states', drawbounds=True, linewidth=1.5)
    #for info, shape in zip(map.states_info, map.states):
        #print info['NAME']
    #    if info['NAME'] == 'Florida':

            #outAr = np.array(shape)
            #np.savetxt('c:\\users\\dwright\\desktop\\' + info['NAME'] + 'testFile.csv',outAr, delimiter=',')
            #sys.exit()


    # I can also find the center of the county and name it!
    # will need to figure out eh size of the name

    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color='aqua')
    # draw lat/lon grid lines every 30 degrees.

    title = 'Exposure Heatmap for {0}'.format(clientName)
    plt.title(title)
    outFileName = 'ExposureMap_{0}'.format(clientName)
    outPath = outPath + '{0}.jpg'.format(outFileName)

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
    testShape = shapefile.Reader(shapePathRoot + shapeName)
    # if correct shape, we ar good
    if testShape.shapeType in [0, 1, 3, 5, 8]:
        shapePath = shapePathRoot + shapeName
    # otherwise, check to see if we've converted this already
    elif convertedName in shapeFileList:
        testShape2 = shapefile.Reader(shapePathRoot + convertedName)
        # if the converted is the right type
        if testShape2.shapeType in [0, 1, 3, 5, 8]:
            shapePath = shapePathRoot + convertedName
    # so we need to convert
    else:
        shapePath = convertShapefile(shapePathRoot + shapeName)
    # clean up the extensions
    shapePath = shapePath.replace('.shp', '')
    return shapePath


def getShapeFile(state, county, path):

    outPath = path + '_working_state'
    # is it not possible for me to pass a shapefile around without
    # writing it to disk? come on!

    return outPath

def getBoundaries(read_path, counties, stateCodes):
    stateCodes = [i for i in stateCodes if i not in ['AK','HI']]

    print 'warning: we are restricting this to only negatives longitudes!'
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
            # we are restricting this to negative longitudes

            thisLon, thisLat = np.amin(shpeAr, 0)
            if thisLon < 0:
              if minLon == 0 or minLon > thisLon:
                  minLon = thisLon
              if minLat == 0 or minLat > thisLat:
                  minLat = thisLat

            thisLon, thisLat = np.amax(shpeAr, 0)
            if thisLon < 0:
              if maxLon == 0 or maxLon < thisLon:
                  maxLon = thisLon
              if maxLat == 0 or maxLat < thisLat:
                  maxLat = thisLat

    return minLat - 1, minLon - 1, maxLat+ 1, maxLon+ 1

def convertShapefile(read_path):
    print 'converting shapefile', read_path

    sf = shapefile.Reader(read_path)
    tp = sf.shapeType
    #print tp

    shapeRecs = sf.records()
    shapeShapes = sf.shapes()
    shapeFields = sf.fields
    #print sf.fields


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
    print 'old type:', tp, 'new type:', to_2D_dict[tp]
    newtp = to_2D_dict[tp]
    wrtr = shapefile.Writer(shapeType=newtp)
    write_path = read_path + '_converted'
    #print len(shapeRecs), len(shapeShapes)

    shapeIndices = range(len(shapeRecs))
    # loop through the shapefiles and create a 2D version of the shapefile we want
    for f in shapeFields:
        #print f
        #wrtr.field(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8])
        # for each type of field, we need to pass them individually, not as a list.. ugh
        wrtr.field(f[0], f[1], f[2], f[3])
    for i in shapeIndices:
        #if i % 10 == 0:
        #    print i - len(shapeRecs)
        shpe = shapeShapes[i].points
        shpts = shapeShapes[i].parts
        rec = shapeRecs[i]
        #print 'points:', shpe

        #print 'new type:', newtp
        if shpts[0] == 0:
            pts = [shpe]
        else:
            pts = shpts
        # we need to pass the shape points as parts with a [] around it because
        # it is expected another nested list
        # in the future we can detect separate shapes like cities within a county
        # and pass them as separate parts in a single shape...
        wrtr.poly(shapeType=newtp, parts=pts)
        wrtr.record(rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7], rec[8])
    print 'writing to', write_path
    wrtr.save(write_path)

    return write_path

def getData(fname, bottomPercentile, topPercentile, p):

    baseData = pd.DataFrame.from_csv(p + fname, index_col=False)

    origValues = baseData.ix[:, 0]
    #print baseData
    #print baseData.ix[:,0]
    midPoint = 0.5
    minVal = baseData.ix[:, 0].astype('float').quantile(bottomPercentile)
    maxVal = baseData.ix[:, 0].astype('float').quantile(topPercentile)

    targetCounties = dict()
    locs = baseData.shape[0]
    for i in range(locs):
        stateName = str(baseData.ix[i, 1]).upper()
        countyName = str(baseData.ix[i, 2]).replace(' COUNTY', '').title()
        targetCounties[stateName + '_' + countyName] = origValues.ix[i, 0]
        #print origValues.ix[i, 0]

    states = baseData.ix[:, 1].unique()


    return states, targetCounties, minVal, maxVal

def main():

    bottomPercentile = 0
    midPoint = 0.5
    topPercentile = 1
    p = '\\'.join(os.path.abspath(__file__).replace('\\','/').split('/')[:-1]) + '\\'
    p = '\\'.join(os.path.abspath(__file__).replace('\\','/').split('/')[:-2]) + '\\'
    print p

    #shapePathRoot = 'C:\\Users\\dwright\\code\\geo_shapes\\'
    shapePathRoot = p + 'geo_shapes\\'
    #p = 'c:\\users\\dwright\\dropbox\\'
    #p = '\\\\BA-FS-NY\Data\\Yr 2016\\Orchid\\Spinnaker Project\\Cat Modeling\\'
    outPath = p + 'output_files\\'
    p = p + 'data_files\\'
    preTxt = ''
    for i in os.listdir(p):
      preTxt = preTxt + i.replace('Exposure.csv','') + '\n'

    clientList = [i.replace('Exposure.csv','') for i in os.listdir(p) if '.csv' in i]
    clist = ' '.join(clientList)

    inptTxt = '\nThe following datasets are available:\n' + preTxt + "\nType in names of datasets from list above separated by space and hit 'enter'.\nIf blank we will use: {0}\n\n".format(clist)
    inpt = raw_input(inptTxt)
    if inpt != '':
        clientList = inpt.split(' ')
    print 'running', clientList
    for clientName in clientList:

        fname = '{0}Exposure.csv'.format(clientName)

        print 'getting data for', clientName
        targetStates, targetCounties, minVal, maxVal = getData(fname, bottomPercentile, topPercentile, p)
        print 'mapping', clientName
        mapping(targetStates, targetCounties, minVal, maxVal, clientName, outPath, shapePathRoot)

if __name__ == '__main__':
    main()