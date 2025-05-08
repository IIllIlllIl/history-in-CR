import reader

# test
# acc = reader.CsvReader("full_info.csv").extract_raw_data().average()["acceptance"]

# test task mapping
tm = reader.TaskMapping("tasks.csv")
print(tm.get_task_id(5, 1))
print(tm.get_task_id(19, 2))

# test csv reader with task mapping
cr = reader.CsvReader("full_info.csv")
rw = cr.extract_raw_data("tasks.csv")
rw.display()


# a_vec = [0] * 3
# for a in acc:
#     if a == -1:
#         continue
#     elif a < 0.2:
#         a_vec[0] += 1
#     elif a < 0.75:
#         a_vec[1] += 1
#     else:
#         a_vec[2] += 1
#
# print(acc)
# print(a_vec)

