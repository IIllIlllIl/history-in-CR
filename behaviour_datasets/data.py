import reader
from scipy import stats
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt


class Data:
    def __init__(self, path):
        self.rd = reader.CsvReader(path).extract_raw_data("tasks.csv")
        self.participants = []
        # offset task difficulty and gender bias
        average = self.rd.average()
        # recording process
        visited_id = 0
        for p in self.rd.participants:
            if p.id > visited_id:
                visited_id = p.id
                self.participants.append(reader.Participant(visited_id, []))
            # minus average time
            for ans in p.answers:
                task_id = ans["task"]
                ave_rt = average["response_time"][task_id]
                if ave_rt > -1:
                    self.participants[-1].add_answer(ans["time"] / ave_rt, round(ans["accept"]), task_id)

    # def check_normality(self, alpha=0.05):
    #     rt_list = self.rd.summary()["response_time"]
    #     for i in range(120):
    #         if len(rt_list[i]) > 3:
    #             stat, p = stats.shapiro(rt_list[i])
    #             print(str(i) + f": p value: {p:.4f}", "-> yes" if p > alpha else "-> no")
    #         else:
    #             print(str(i) + ": Data must be at least length 3.")

    def display(self):
        for p in self.participants:
            p.display()

    def response_time(self):
        rt = []
        for p in self.participants:
            temp_rt_row = []
            for ans in p.answers:
                temp_rt_row.append(ans["time"])
            rt.append(temp_rt_row)
        return rt

    def acceptance(self):
        acc = []
        for p in self.participants:
            temp_rt_row = []
            for ans in p.answers:
                temp_rt_row.append(ans["accept"])
            acc.append(temp_rt_row)
        return acc


class Analysis:
    def __init__(self, data):
        self.data = data
        self.eff = []
        for row in self.data:
            temp_eff = []
            for time in row:
                temp_eff.append(1 / time)
            self.eff.append(temp_eff)

    def check_normality(self, alpha=0.05):
        cnt_y = 0
        cnt_n = 0
        for row in self.data:
            if len(row) > 3:
                stat, p = stats.shapiro(row)
                if p > alpha:
                    print(f"p value: {p:.4f}", "-> yes")
                    cnt_y += 1
                else:
                    print(f"p value: {p:.4f}", "-> no")
                    cnt_n += 1
            else:
                print("Data must be at least length 3.")
                cnt_n += 1
        return [cnt_y, cnt_n]

    def display(self):
        for row in self.eff:
            print(row)

    def pelt_check_point(self):
        def linear_regression_cost(data, start, end):
            segment = data[start:end]
            x_segment = np.arange(len(segment))
            # 拟合线性模型
            slope, intercept = np.polyfit(x_segment, segment, 1)
            # 计算残差平方和
            residuals = segment - (slope * x_segment + intercept)
            return np.sum(residuals ** 2)
        model = rpt.Pelt(custom_cost=linear_regression_cost)
        temp_y = np.array(self.data[0])
        print(temp_y)
        change_points = model.fit_predict(temp_y, pen=5)
        print("检测到的斜率突变点位置:", change_points[:-1])

        plt.plot(temp_y, label="原始数据")
        for cp in change_points:
            plt.axvline(x=cp, color='r', linestyle='--', label='突变点')
        plt.show()

    def slt_decomposition(self):
        for i in range(len(self.eff)):
            temp_y = np.array(self.eff[i])
            if temp_y.size < 45:
                continue
            # stl
            decomposition = STL(temp_y, period=15).fit()
            decomposition.plot()
            plt.savefig("result/slt_" + str(i) + ".png")
            # plt.show()

    def get_history_pars(self, pid, window_size):
        participant_eff = self.eff[pid]
        if len(participant_eff) < window_size:
            return [[], [], [], []]
        # history pars
        previous_eff = []
        longer_eff = []
        streak = []
        for i in range(window_size - 1, len(participant_eff)):
            previous_eff.append(participant_eff[i - 1])
            longer_eff.append(sum(participant_eff[i-window_size+1:i-1]))
            temp_streak = 0
            his_eff = participant_eff[i-window_size+1:i]
            his_eff.reverse()
            for his in his_eff:
                if his > 1:
                    temp_streak += 1
                else:
                    break
            streak.append(temp_streak)
        return [participant_eff[window_size - 1:], previous_eff, longer_eff, streak]

    def get_ols_result(self, pid, window_size):
        # generate X & y
        eff, p_eff, l_eff, s = self.get_history_pars(pid, window_size)
        t = range(len(eff))
        # differentiate streak
        streak = []
        for cnt in s:
            temp_s = []
            for i in range(window_size):
                if i == cnt:
                    temp_s.append(1)
                else:
                    temp_s.append(0)
            streak.append(temp_s)
        y = np.array(eff)
        x_vec = np.column_stack((t, p_eff, l_eff, streak))
        x_vec = sm.add_constant(x_vec)
        # ols
        model = sm.OLS(y, x_vec)
        results = model.fit()
        # output
        # print(results.summary())
        coefficients = results.params[1:]
        p_values = results.pvalues[1:]
        # x_label = ["index", "eff(i-1)", "eff(window)", "streak"]
        # for i in range(len(coefficients)):
        #     print(f"{x_label[i]}: {coefficients[i]:.4f}, p: {p_values[i]:.4f}")
        return [coefficients, p_values]

    def ols_regression(self, window_size=6):
        coef_buffer = [[] for _ in range(window_size + 3)]
        pid_buffer = [[] for _ in range(window_size + 3)]
        for pid in range(len(self.eff)):
            coef, p_value = self.get_ols_result(pid, window_size)
            for i in range(window_size + 3):
                if p_value[i] < 0.05:
                    coef_buffer[i].append(coef[i])
                    pid_buffer[i].append(pid)
        for c_vec in coef_buffer:
            print(c_vec)
        for p_vec in pid_buffer:
            print(p_vec)
        for c_vec in coef_buffer:
            print(f"average: {sum(c_vec) / len(c_vec):.4f}, cnt: {len(c_vec)}")

    def entire_ols_result(self, window_size=6):
        # generate X & y
        t_list = []
        pre_list = []
        his_list = []
        str_list = []
        y_list = []
        # sum up
        for pid in range(len(self.eff)):
            eff, p_eff, l_eff, s = self.get_history_pars(pid, window_size)
            t = range(len(eff))
            t_list += t
            # differentiate streak
            streak = []
            for cnt in s:
                temp_s = []
                for i in range(window_size):
                    if i == cnt:
                        temp_s.append(1)
                    else:
                        temp_s.append(0)
                streak.append(temp_s)
            y_list += eff
            pre_list += p_eff
            his_list += l_eff
            str_list += streak
        x_vec = np.column_stack((t_list, pre_list, his_list, str_list))
        x_vec = sm.add_constant(x_vec)
        # ols
        model = sm.OLS(np.array(y_list), x_vec)
        results = model.fit()
        # output
        coefficients = results.params[1:]
        p_values = results.pvalues[1:]
        for coef, p in zip(coefficients, p_values):
            print(f"coef: , {coef:.5f}, \tp_value:  , {p:.6f}")
        return [coefficients, p_values]


class Streaks:
    def __init__(self, data):
        self.data = data
        # print(data)

    def display(self):
        for row in self.data:
            print(row)

    @staticmethod
    def get_matrix(row):
        # i\i + 1  0   1   2
        #   0      00  10  20
        #   1      01  11  21
        #   2      02  12  22
        temp_matrix = [[0 for _ in range(3)] for _ in range(3)]
        for i in range(len(row) - 1):
            temp_matrix[row[i]][row[i + 1]] += 1
        return temp_matrix

    # def one_way_anova(self):
    #     for row in self.data:
    #         print(row)
    #         matrix = self.get_history_2(row)
    #         print(matrix)
    #         group00 = np.array(matrix[0])
    #         group01 = np.array(matrix[1])
    #         group10 = np.array(matrix[2])
    #         group11 = np.array(matrix[3])
    #         f_stat, p_value = stats.f_oneway(group00, group01, group10, group11)
    #         print(f"F统计量 = {f_stat:.4f}")
    #         print(f"P值 = {p_value:.4f}")

    def one_previous_anova_each(self):
        max_p = 0
        min_p = 1
        for row in self.data:
            # print(row)
            matrix = self.get_matrix(row)
            # print(matrix)
            group0 = np.array(matrix[0])
            group1 = np.array(matrix[1])
            group2 = np.array(matrix[2])
            f_stat, p_value = stats.f_oneway(group0, group1, group2)
            max_p = max(max_p, p_value)
            min_p = min(min_p, p_value)
            print(f"F统计量 = {f_stat:.4f}")
            print(f"P值 = {p_value:.4f}")
        print(f"{min_p:.4f} - {max_p:.4f}")

    def one_previous_anova(self):
        matrix = np.zeros((3, 3))
        for row in self.data:
            temp_matrix = np.array(self.get_matrix(row))
            matrix += temp_matrix
        print(matrix)
        group0 = np.array(matrix[0, :])
        group1 = np.array(matrix[1, :])
        group2 = np.array(matrix[2, :])
        f_stat, p_value = stats.f_oneway(group0, group1, group2)
        print(f"F统计量 = {f_stat:.4f}")
        print(f"P值 = {p_value:.4f}")

    @staticmethod
    def adjusted_residual(observed, expected):
        row_sums = observed.sum(axis=1)
        col_sums = observed.sum(axis=0)
        total = observed.sum()
        row_proportions = row_sums / total
        col_proportions = col_sums / total
        adjusted_residuals = np.zeros_like(observed, dtype=float)
        for i in range(observed.shape[0]):
            for j in range(observed.shape[1]):
                e = expected[i, j]
                # 调整残差分母
                denominator = np.sqrt(e * (1 - row_proportions[i]) * (1 - col_proportions[j]))
                # 调整残差
                adjusted_residuals[i, j] = (observed[i, j] - e) / denominator
        print(np.round(adjusted_residuals, 2))

    def phi_contingency(self):
        matrix = np.zeros((3, 3))
        for row in self.data:
            # print(row)
            temp_matrix = np.array(self.get_matrix(row))
            matrix += temp_matrix
        print(matrix)
        chi2, p, dof, expected = stats.chi2_contingency(matrix)
        n = matrix.sum()
        phi = np.sqrt(chi2 / n)
        print("phi: " + str(phi) + "\tp: " + str(p))
        self.adjusted_residual(matrix, expected)

    # def phi_contingency_each(self):
    #     max_phi = 0
    #     max_p = 0
    #     min_phi = 10
    #     min_p = 1
    #     for row in self.data:
    #         # print(row)
    #         matrix = np.array(self.get_matrix(row))
    #         # print(matrix)
    #         chi2, p, dof, expected = stats.chi2_contingency(matrix)
    #         n = matrix.sum()
    #         phi = np.sqrt(chi2 / n)
    #         max_p = max(max_p, p)
    #         min_p = min(min_p, p)
    #         max_phi = max(max_phi, phi)
    #         min_phi = min(min_phi, phi)
    #         if p < 0.05:
    #             print("* phi: " + str(phi) + "\tp: " + str(p))
    #     print(f"phi: {min_phi:.4f}, {max_phi:.4f}")
    #     print(f"p: {min_p:.4f}, {max_p:.4f}")

    @staticmethod
    def get_matrix_2(row):
        temp_matrix = [[0 for _ in range(3)] for _ in range(9)]
        for i in range(len(row) - 2):
            temp_matrix[row[i] * 3 + row[i + 1]][row[i + 2]] += 1
        return temp_matrix

    def two_previous_anova_each(self):
        max_f = 0
        max_p = 0
        min_f = 10
        min_p = 1
        for row in self.data:
            # print(row)
            matrix = self.get_matrix_2(row)
            # print(matrix)
            group = []
            for i in range(9):
                group.append(np.array(matrix[i]))
            f_stat, p_value = stats.f_oneway(group[0], group[1], group[2], group[3], group[4],
                                             group[5], group[6], group[7], group[8])
            max_p = max(max_p, p_value)
            min_p = min(min_p, p_value)
            max_f = max(max_f, f_stat)
            min_f = min(min_f, f_stat)
            if p_value <= 0.05:
                print(f"F统计量 = {f_stat:.4f}")
                print(f"P值 = {p_value:.4f}")
        print(f"f: {min_f:.4f}, {max_f:.4f}")
        print(f"p: {min_p:.4f}, {max_p:.4f}")

    def two_previous_anova(self):
        matrix = np.zeros((9, 3))
        for row in self.data:
            # print(row)
            temp_matrix = self.get_matrix_2(row)
            matrix += temp_matrix
        print(matrix)
        group = []
        for i in range(9):
            group.append(np.array(matrix[i]))
        f_stat, p_value = stats.f_oneway(group[0], group[1], group[2], group[3], group[4],
                                         group[5], group[6], group[7], group[8])
        print(f"F统计量 = {f_stat:.4f}")
        print(f"P值 = {p_value:.4f}")

    def two_previous_chi2(self):
        matrix = np.zeros((9, 3))
        for row in self.data:
            # print(row)
            temp_matrix = self.get_matrix_2(row)
            matrix += temp_matrix
        print(matrix)
        chi2, p, dof, expected = stats.chi2_contingency(matrix)
        n = matrix.sum()
        phi = np.sqrt(chi2 / n)
        print(f"phi: {phi:.4f}")
        print(f"p: {p}")
        self.adjusted_residual(matrix, expected)

    # def two_previous_chi2_each(self):
    #     max_phi = 0
    #     max_p = 0
    #     min_phi = 10
    #     min_p = 1
    #     for row in self.data:
    #         # print(row)
    #         matrix = np.array(self.get_matrix_2(row))
    #         print(matrix)
    #         chi2, p, dof, expected = stats.chi2_contingency(matrix)
    #         n = matrix.sum()
    #         phi = np.sqrt(chi2 / n)
    #         max_p = max(max_p, p)
    #         min_p = min(min_p, p)
    #         max_phi = max(max_phi, phi)
    #         min_phi = min(min_phi, phi)
    #         if p < 0.05:
    #             print("* phi: " + str(phi) + "\tp: " + str(p))
    #     print(f"phi: {min_phi:.4f}, {max_phi:.4f}")
    #     print(f"p: {min_p:.4f}, {max_p:.4f}")

    def get_history_pars(self, pid, window_size=6):
        participant = self.data[pid]
        if len(participant) < window_size:
            return [[], [], [], []]
        # history pars
        previous = []
        longer = []
        streak = []
        for i in range(window_size - 1, len(participant)):
            previous.append(participant[i - 1])
            longer.append(participant[i-window_size+1:i-1].count(1))
            temp_streak = 0
            his_eff = participant[i-window_size+1:i]
            his_eff.reverse()
            for his in his_eff:
                if his == 1:
                    temp_streak += 1
                else:
                    break
            streak.append(temp_streak)
        return [participant[window_size - 1:], previous, longer, streak]

    def entire_ols_result(self, window_size=6):
        # generate X & y
        t_list = []
        pre_list = []
        his_list = []
        str_list = []
        y_list = []
        # sum up
        for pid in range(len(self.data)):
            eff, p_eff, l_eff, s = self.get_history_pars(pid, window_size)
            t = range(len(eff))
            t_list += t
            # differentiate streak
            streak = []
            for cnt in s:
                temp_s = []
                for i in range(window_size):
                    if i == cnt:
                        temp_s.append(1)
                    else:
                        temp_s.append(0)
                streak.append(temp_s)
            # print(self.data[pid])
            # print(eff)
            # print(p_eff)
            # print(l_eff)
            # print(streak)
            y_list += eff
            pre_list += p_eff
            his_list += l_eff
            str_list += streak
        x_vec = np.column_stack((t_list, pre_list, his_list, str_list))
        x_vec = sm.add_constant(x_vec)
        # ols
        model = sm.OLS(np.array(y_list), x_vec)
        results = model.fit()
        # output
        coefficients = results.params[1:]
        p_values = results.pvalues[1:]
        for coef, p in zip(coefficients, p_values):
            print(f"coef: , {coef:.5f}, \tp_value:  , {p:.6f}")
        return [coefficients, p_values]


# run analysis
dt = Data("full_info.csv")
a = Analysis(dt.response_time())
st = Streaks(dt.acceptance())

# trend
# a.slt_decomposition()

# linear model
# a.ols_regression()
# a.entire_ols_result()

# anova
# st.one_previous_anova()
# st.one_previous_anova_each()
# st.two_previous_anova()
# st.two_previous_anova_each()

# chi2
st.phi_contingency()
st.two_previous_chi2()

# OLS streak
# st.entire_ols_result()
