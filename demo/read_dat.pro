shape = [99, 400]
Te_data = read_binary('Te.dat', data_type=4, data_dims=shape)
Rmaj_data = read_binary('Rmaj.dat', data_type=4, data_dims=shape)

plt = plot(Rmaj_data[4, *], Te_data[4, *])