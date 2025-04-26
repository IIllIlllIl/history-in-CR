import data

# test data
dt = data.Data("full_info.csv")
dt.display()
print(dt.response_time())

# test analysis
a = data.Analysis(data.Data("full_info.csv").response_time())
print(a.check_normality())
a.display()
a.pelt_check_point()
a.slt_decomposition()
a.get_history_pars(0)
