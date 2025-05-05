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


# run analysis
a = Analysis(Data("full_info.csv").response_time())
# trend
a.slt_decomposition()
# linear model
a.ols_regression()
