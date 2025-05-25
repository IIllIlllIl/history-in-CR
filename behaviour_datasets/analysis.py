import data

# run analysis
dt = data.Data("full_info.csv")
a = data.Analysis(dt.response_time())
st = data.Streaks(dt.acceptance())

# trend
# a.slt_decomposition()
a.anova_sum_of_group()
a.repeated_measures_anova()

# linear model
# a.ols_regression()
# a.entire_ols_result()

# anova
# st.one_previous_anova()
# st.one_previous_anova_each()
# st.two_previous_anova()
# st.two_previous_anova_each()

# chi2
# st.phi_contingency()
# st.two_previous_chi2()

# OLS streak
# st.entire_ols_result()
