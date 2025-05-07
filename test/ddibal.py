import sys
print(sys.executable)
import os
print(os.getcwd())

from tabulate import tabulate

table = [["이름", "나이"], ["홍길동", 25], ["김철수", 30]]
print(tabulate(table, headers="firstrow", tablefmt="grid"))