from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import numpy as np
import shapefile
import sys


def main():

    # set up orthographic map projection with
    # perspective of satellite looking down at 50N, 100W.
    # use low resolution coastlines.

    fig = plt.figure()
    ax = fig.add_subplot(111)

    map = Basemap(llcrnrlon=-80.,llcrnrlat=35.,urcrnrlon=-70.,urcrnrlat=42.,
                  projection='aeqd', lat_0=37.533, lon_0=-77.46, resolution='l')
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='green', lake_color='aqua')
    #map.drawstates(linewidth=.25)
    #map.drawcounties(linewidth=.1)

    shapepath = 'C:\\Users\\dwright\\code\\county_files\\cb_2014_us_county_20m'

    targetStates = ['VA']
    targetCounties = {'Accomack': 'red', 'Amelia': 'yellow'}

    testShape = shapefile.Reader(shapepath)
    tp = testShape.shapeType
    if tp not in [0, 1, 3, 5, 8]:
        shapepath = convertShapefile(shapepath)

    #for targetState in targetStates:
    #    for targetCounty in targetCounties.keys():
            #statePath = getShapeFile(targetState, targetCounty, shapepath)
    patches = []
    col = targetCounties[targetCounty]
    map.readshapefile(shapepath, 'counties', drawbounds=True)
    for info, shape in zip(map.counties_info, map.counties):
        if info['nombre'] == 'Accomack':
            patches.append(Polygon(np.array(shape), True))
            ax.add_collection(PatchCollection(patches, facecolor=col, edgecolor='black',
                                              linewidths=1., zorder=2))
            # google "filling shapefile polygons"
            plt.show()
            sys.exit()
    # I can also find the center of the county and name it!
    # will need to figure out eh size of the name

    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color='aqua')
    # draw lat/lon grid lines every 30 degrees.


    # compute native map projection coordinates of lat/lon grid.

    #x, y = map(lons*180./np.pi, lats*180./np.pi)
    # contour data over the map.

    #cs = map.contour(x,y,wave+mean,15,linewidths=1.5)
    plt.title('contour lines over filled continent background')
    plt.show()

    return


def getShapeFile(state, county, path):

    outPath = path + '_working_state'
    # is it not possible for me to pass a shapefile around without
    # writing it to disk? come on!



    return outPath

def convertShapefile(read_path):

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


if __name__ == '__main__':
    main()