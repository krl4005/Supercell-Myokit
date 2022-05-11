import myokit.formats

i = myokit.formats.importer('cellml')
mod = i.model('./Ohara_Rudy_2011.cellml')

myokit.save('ord.mmt', mod)
