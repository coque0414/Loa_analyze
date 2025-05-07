import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

plt.rc('font', family='Malgun Gothic')  
plt.rc('axes', unicode_minus=False)

# 데이터 설정
dates = ["4.23", "4.24", "4.25", "4.26", "4.27", "4.28", "4.29", "4.30", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6"]
sales_counts = [550, 600, 550, 580, 800, 500, 350, 7000, 4000, 1200, 800, 900, 700, 300]
avg_prices = [133766.7, 132806.3, 135828.1, 138449.5, 138742.9, 139169.6, 138562.9,
              88084.4, 87604.5, 88390, 88217.8, 86910.2, 83283.2, 80962.4]

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(10, 6))

# 판매 건수 바 차트 (왼쪽 y축)
color_bar = "#2196F3"
bars = ax1.bar(dates, sales_counts, color=color_bar, label="판매 건수")
ax1.set_ylabel("판매 건수", color=color_bar)
ax1.tick_params(axis='y', labelcolor=color_bar)
ax1.set_ylim(0, 8000)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(2000))

# 평균 거래가 라인 차트 (오른쪽 y축)
ax2 = ax1.twinx()
color_line = "#00C853"
line = ax2.plot(dates, avg_prices, marker='o', color=color_line, label="평균 거래가")
ax2.set_ylabel("평균 거래가", color=color_line)
ax2.tick_params(axis='y', labelcolor=color_line)
ax2.set_ylim(80000, 160000)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(20000))

# 데이터 라벨 추가
for x, y in zip(dates, avg_prices):
    ax2.text(x, y, f"{y:.1f}", fontsize=8, ha='center', va='bottom', backgroundcolor="#00C853")

# 레전드 추가
lines_labels = [bars, line[0]]
ax1.legend(lines_labels, [l.get_label() for l in lines_labels], loc='upper right')

# 그리드 및 레이아웃 설정
ax1.grid(axis='y', linestyle='--', alpha=0.5)
plt.title("판매 건수 및 평균 거래가 추이")
plt.xticks(rotation=0)
plt.tight_layout()

# 그래프 출력
plt.show()
