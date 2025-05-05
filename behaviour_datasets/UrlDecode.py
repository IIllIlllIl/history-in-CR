import csv


class Url:
    def __init__(self, path):
        self.target_columns = ["pull_request_file_name"]
        with open(path, "r", encoding="utf-8") as file:
            self.reader = csv.DictReader(file)
            self.selected_data = []
            for row in self.reader:
                selected_row = {col: row[col] for col in self.target_columns}
                self.selected_data.append(selected_row)
            # print(self.selected_data)

    def decode(self):
        index = 2
        for line in self.selected_data:
            link = line["pull_request_file_name"][:-4]
            link = link.replace("_", "/")
            # print(str(index) + ": " + link)
            print(link)
            index += 1


u = Url("tasks.csv")
u.decode()
