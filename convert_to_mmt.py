import myokit.formats

i = myokit.formats.importer('cellml')
mod = i.model('./Tomek_model_endo.cellml')

myokit.save('tor_ord.mmt', mod)
