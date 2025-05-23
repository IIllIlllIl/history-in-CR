import csv
import matplotlib.pyplot as plt


# reading csv data
class CsvReader:
    def __init__(self, path):
        self.target_columns = ["PID", "RES_DURATION", "RES_HAND", "PNG_NAME", "AUTHOR_GENDER", "VERSION"]
        with open(path, "r", encoding="utf-8") as file:
            self.reader = csv.DictReader(file)
            self.selected_data = []
            for row in self.reader:
                selected_row = {col: row[col] for col in self.target_columns}
                self.selected_data.append(selected_row)
            # print(self.selected_data)

    def remove_na_row(self, remove=True, threshold=15, plot=False):
        remove_list = []
        na_pid = {}
        na_cnt = 0
        for row in self.selected_data:
            if 'N/A' in row.values():
                remove_list.append(row)
            elif "NA" in row.values():
                if row["PID"] in na_pid.keys():
                    na_pid[row["PID"]] += 1
                else:
                    na_pid[row["PID"]] = 1
                na_cnt += 1
                row["RES_DURATION"] = 30000.0
        # remove participants with NA not less than threshold
        if remove:
            pid_list = []
            for [k, v] in na_pid.items():
                if v >= threshold:
                    pid_list.append(k)
            print(f"removed pid: {pid_list}")
            for row in self.selected_data:
                if row["PID"] in pid_list and 'N/A' not in row.values():
                    remove_list.append(row)
        for row in remove_list:
            self.selected_data.remove(row)
        # plot NA of each participant
        # print(self.selected_data)
        if plot:
            cnt = [0 for _ in range(9)]
            for [k, v] in na_pid.items():
                print(f"{k}: {v}")
                cnt[int(v/5)] += 1
            self.na_plot(cnt)

    @staticmethod
    def na_plot(data):
        x_labels = [f"[{i * 5}, {i* 5 + 5})" for i in range(len(data))]
        plt.figure(figsize=(10, 6))  # 设置图表大小（可选）
        bars = plt.bar(x_labels, data, color='skyblue', edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        plt.title("time out frequency of each participant", fontsize=14)
        plt.xlabel("Categories", fontsize=12)
        plt.ylabel("Values", fontsize=12)
        plt.xticks(rotation=45)  # 旋转x轴标签避免重叠
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加横向网格线
        plt.tight_layout()  # 自动调整布局
        plt.savefig("result/pid_na.png")
        # plt.show()

    # change selected data to raw data
    def extract_raw_data(self, task_path):
        self.remove_na_row()
        participant_list = []
        # recording process
        visited_id = 0
        for row in self.selected_data:
            if int(row["PID"]) > visited_id:
                visited_id = int(row["PID"])
                participant_list.append(Participant(visited_id, []))
            # change response_time into "s"
            response = float(row["RES_DURATION"]) / 1000
            # digitalizing acceptance
            if row["RES_HAND"] == "L":
                acc = 1
            elif row["RES_HAND"] == "R":
                acc = 0
            else:
                acc = 2
            # digitalizing task id
            tm = TaskMapping(task_path)
            task_id = tm.get_task_id(int(row["PNG_NAME"].split(".")[0]), int(row["VERSION"]))
            # task_id = int(row["PNG_NAME"].split(".")[0])
            # if row["AUTHOR_GENDER"] == "M":
            #     task_id += 60
            # elif row["AUTHOR_GENDER"] == "C":
            #     task_id += 120
            participant_list[-1].add_answer(response, acc, task_id)
        return RawData(participant_list)


class TaskMapping:
    def __init__(self, path):
        self.target_columns = ["v1_stimuli_name", "v2_stimuli_name"]
        with open(path, "r", encoding="utf-8") as file:
            self.reader = csv.DictReader(file)
            self.selected_data = []
            for row in self.reader:
                selected_row = {col: row[col] for col in self.target_columns}
                self.selected_data.append(selected_row)
            # print(self.selected_data)
        # reform data
        task_cnt = len(self.selected_data)
        self.v1_tasks = [0 for _ in range(task_cnt)]
        self.v2_tasks = [0 for _ in range(task_cnt)]
        for i in range(task_cnt):
            self.v1_tasks[int(self.selected_data[i]["v1_stimuli_name"].split(".")[0])] = i
            self.v2_tasks[int(self.selected_data[i]["v2_stimuli_name"].split(".")[0])] = i
        # print(self.v1_tasks)
        # print(self.v2_tasks)

    def get_task_id(self, png_id, version):
        if version == 1:
            return self.v1_tasks[png_id]
        elif version == 2:
            return self.v2_tasks[png_id] + 60
        else:
            return -1


# recording behaviours of each participant
class Participant:
    def __init__(self, pid, answer_list):
        self.id = pid
        self.answers = answer_list

    def add_answer(self, response_time, acceptance, task_id):
        self.answers.append({"time": response_time, "accept": acceptance, "task": task_id})

    def display(self):
        print(self.id)
        print(self.answers)


# original data
class RawData:
    def __init__(self, participant_list):
        self.participants = participant_list

    def display(self):
        for p in self.participants:
            p.display()

    def average(self, task_length=120):
        sum_rt = [0] * task_length
        sum_acc = [0] * task_length
        cnt_task = [0] * task_length
        for p in self.participants:
            for ans in p.answers:
                cnt_task[ans["task"]] += 1
                sum_rt[ans["task"]] += ans["time"]
                sum_acc[ans["task"]] += ans["accept"]
        ave_rt = []
        ave_acc = []
        for i in range(task_length):
            if cnt_task[i] > 0:
                ave_rt.append(sum_rt[i] / cnt_task[i])
                ave_acc.append(sum_acc[i] / cnt_task[i])
            else:
                ave_rt.append(-1)
                ave_acc.append(-1)
        return {"response_time": ave_rt, "acceptance": ave_acc}

    def summary(self, task_length=120):
        sum_rt = [[] for _ in range(task_length)]
        sum_acc = [[] for _ in range(task_length)]
        cnt_task = [0] * task_length
        for p in self.participants:
            for ans in p.answers:
                cnt_task[ans["task"]] += 1
                sum_rt[ans["task"]].append(ans["time"])
                sum_acc[ans["task"]].append(ans["accept"])
        return {"response_time": sum_rt, "acceptance": sum_acc, "count": cnt_task}

