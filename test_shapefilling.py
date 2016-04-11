import shapefile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.cm as cm

#   -- input --

path = 'c:\\users\\dwright\\code\\geo_shapes\\'
shpName = 'cb_2014_us_county_500k'
sf = shapefile.Reader(path + shpName)
recs = sf.records()
shapes = sf.shapes()
Nshp = len(shapes)
cns = []
for nshp in xrange(Nshp):
    cns.append(recs[nshp][1])
cns = np.array(cns)
cmap = cm.get_cmap('Dark2')
vals = 1.* np.arange(Nshp)/Nshp
cccol = cmap(vals)
#   -- plot --
fig = plt.figure()
ax = fig.add_subplot(111)
cols = []
print 'looping through shapefile'

c = Nshp
for nshp in xrange(Nshp):
    if c % 100 == 0:
        print c, recs[nshp][5]
    ptchs = []
    pts = np.array(shapes[nshp].points)
    prt = shapes[nshp].parts
    par = list(prt) + [pts.shape[0]]
    #print recs[nshp][5]
    for pij in xrange(len(prt)):
     ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
    pcol = PatchCollection(ptchs,facecolor=cccol[nshp,:],edgecolor='k', linewidths=.1)
    ax.add_collection(pcol)
pcol.set_array(vals)

ax.set_xlim(-120,-60)
ax.set_ylim(20,90)


#fig.savefig('test.png')
plt.show()


# I need to see if I can get a colorbar in here
# to do that I need to find the mappable variable here and set it to the colorbar.. AX doesn't work

# I want to test if it is the colorbar in the other one that is givine me trouble..
#