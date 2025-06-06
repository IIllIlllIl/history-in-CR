import data


# test data
def test_data():
    dt = data.Data("full_info.csv")
    dt.display()


# test analysis
def test_analysis():
    a = data.Analysis(data.Data("full_info.csv").response_time())
    # print(a.check_normality())
    a.display()
    a.pelt_check_point()
    a.slt_decomposition()
    a.get_history_pars(0)


# test streaks
def test_streaks():
    dt = data.Data("full_info.csv")
    st = data.Streaks(dt.acceptance())
    st.phi_contingency()

    arr = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    print(st.get_matrix(arr))


# test phi
def test_phi():
    dt = data.Data("full_info.csv")
    st = data.Streaks(dt.acceptance())
    st.phi_contingency()


test_data()
