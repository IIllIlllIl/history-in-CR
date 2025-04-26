import csv


# reading csv data
class CsvReader:
    def __init__(self, path):
        self.target_columns = ["PID", "RES_DURATION", "RES_HAND", "PNG_NAME", "AUTHOR_GENDER"]
        with open(path, "r", encoding="utf-8") as file:
            self.reader = csv.DictReader(file)
            self.selected_data = []
            for row in self.reader:
                selected_row = {col: row[col] for col in self.target_columns}
                self.selected_data.append(selected_row)
            # print(self.selected_data)

    def remove_na_row(self):
        na_list = []
        for row in self.selected_data:
            if "NA" in row.values() or 'N/A' in row.values():
                na_list.append(row)
        for row in na_list:
            self.selected_data.remove(row)
        # print(self.selected_data)

    # change selected data to raw data
    def extract_raw_data(self):
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
            else:
                acc = 0
            # digitalizing task id
            task_id = int(row["PNG_NAME"].split(".")[0])
            if row["AUTHOR_GENDER"] == "M":
                task_id += 60
            elif row["AUTHOR_GENDER"] == "C":
                task_id += 120
            participant_list[-1].add_answer(response, acc, task_id)
        return RawData(participant_list)


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

    def average(self):
        sum_rt = [0] * 180
        sum_acc = [0] * 180
        cnt_task = [0] * 180
        for p in self.participants:
            for ans in p.answers:
                cnt_task[ans["task"]] += 1
                sum_rt[ans["task"]] += ans["time"]
                sum_acc[ans["task"]] += ans["accept"]
        ave_rt = []
        ave_acc = []
        for i in range(180):
            if cnt_task[i] > 0:
                ave_rt.append(sum_rt[i] / cnt_task[i])
                ave_acc.append(sum_acc[i] / cnt_task[i])
            else:
                ave_rt.append(-1)
                ave_acc.append(-1)
        return {"response_time": ave_rt, "acceptance": ave_acc}

    def summary(self):
        sum_rt = [[] for _ in range(180)]
        sum_acc = [[] for _ in range(180)]
        cnt_task = [0] * 180
        for p in self.participants:
            for ans in p.answers:
                cnt_task[ans["task"]] += 1
                sum_rt[ans["task"]].append(ans["time"])
                sum_acc[ans["task"]].append(ans["accept"])
        return {"response_time": sum_rt, "acceptance": sum_acc, "count": cnt_task}
