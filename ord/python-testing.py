import OpenCOR as opencor

s = opencor.openRemoteSimulation('https://models.physiomeproject.org/w/andre/SAN-ORd/rawfile/fe51425ce5ef054bcf04128f29e22346717def48/action-potential.xml')
d = s.data()
d.setStartingPoint(0)
d.setEndingPoint(1000)
d.setPointInterval(1)
s.run()

t = 0
while t < 1001:
    next_t = t + 1
    d.setStartingPoint(t)
    d.setEndingPoint(next_t)
    d.setPointInterval(1)
    s.run()
    t = next_t
    