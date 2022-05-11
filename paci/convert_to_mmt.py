import myokit.formats

i = myokit.formats.importer('cellml')
mod = i.model('./paci_ventricularVersion.cellml')

myokit.save('paci2013_ventricular.mmt', mod)
