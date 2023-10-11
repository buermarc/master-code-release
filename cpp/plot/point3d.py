import pandas
mat = pandas.read_csv("../../matlab/file1.csv")
mat = mat[mat.columns[17*3:18*3]]
one = pandas.read_csv("../data/point3d.csv")

breakpoint()
