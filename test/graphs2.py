import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')  
plt.rc('axes', unicode_minus=False)

def plot_sales_and_price(
    dates, sales_counts, avg_prices,
    pred_dates=None, pred_sales=None, pred_avg_prices=None,
    color_bar='#2196F3', color_bar_pred='#90CAF9',
    color_line='#00C853', color_line_pred='#A5D6A7',
    line_style_actual='-', line_style_pred='--'
):
    """
    판매 건수와 평균 거래가를 실제 데이터와 예측 데이터로 구분하여 그립니다.

    params:
        dates (list of str): 실제 데이터 날짜
        sales_counts (list of int): 실제 판매 건수
        avg_prices (list of float): 실제 평균 거래가
        pred_dates (list of str): 예측 데이터 날짜
        pred_sales (list of int): 예측 판매 건수
        pred_avg_prices (list of float): 예측 평균 거래가
        color_bar (str): 실제 판매 건수 막대 색상
        color_bar_pred (str): 예측 판매 건수 막대 색상
        color_line (str): 실제 평균 거래가 선 색상
        color_line_pred (str): 예측 평균 거래가 선 색상
        line_style_actual (str): 실제 평균 거래가 선 스타일
        line_style_pred (str): 예측 평균 거래가 선 스타일
    """
    fig, ax1 = plt.subplots(figsize=(13, 6))

    # 실제 판매 건수 바 차트
    bars = ax1.bar(dates, sales_counts, color=color_bar, label='실제 판매 건수')
    if pred_dates and pred_sales:
        bars_pred = ax1.bar(pred_dates, pred_sales, color=color_bar_pred, alpha=0.7, label='예측 판매 건수')

    ax1.set_ylabel('판매 건수', color=color_bar)
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.set_ylim(0, max(sales_counts + (pred_sales or [0])) * 1.1)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(int(max(sales_counts) * 0.5)))

    # 평균 거래가 라인 차트
    ax2 = ax1.twinx()
    # 실제 구간 실선
    line_actual = ax2.plot(dates, avg_prices, marker='o', linestyle=line_style_actual, color=color_line, label='실제 평균 거래가')
    # 예측 구간과 연결 구간 점선
    if pred_dates and pred_avg_prices:
        # 연결: 마지막 실제 점 to 첫 예측 점
        connect_x = [dates[-1], pred_dates[0]]
        connect_y = [avg_prices[-1], pred_avg_prices[0]]
        ax2.plot(connect_x, connect_y, marker='o', linestyle=line_style_pred, color=color_line_pred)
        # 예측 점선
        line_pred = ax2.plot(pred_dates, pred_avg_prices, marker='o', linestyle=line_style_pred, color=color_line_pred, label='예측 평균 거래가')

    ax2.set_ylabel('평균 거래가', color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)
    combined = avg_prices + (pred_avg_prices or [])
    ax2.set_ylim(min(combined) * 0.9, max(combined) * 1.1)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator((max(combined) - min(combined)) / 5))

    # 데이터 라벨
    for x, y in zip(dates, avg_prices): ax2.text(x, y, f"{y:.1f}", fontsize=8, ha='center', va='bottom', backgroundcolor=color_line)
    if pred_dates and pred_avg_prices:
        for x, y in zip(pred_dates, pred_avg_prices): ax2.text(x, y, f"{y:.1f}", fontsize=8, ha='center', va='bottom', backgroundcolor=color_line_pred)

    # 범례
    handles = [bars, line_actual[0]]
    labels = [h.get_label() for h in handles]
    if pred_dates and pred_sales: handles += [bars_pred]; labels += [bars_pred.get_label()]
    if pred_dates and pred_avg_prices: handles += [line_pred[0]]; labels += [line_pred[0].get_label()]
    ax1.legend(handles, labels, loc='upper right')

    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.title('판매 건수 및 평균 거래가 추이 (실제 vs 예측)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 실제/예측 데이터 정의 및 그래프 호출
if __name__ == '__main__':
    # 실제 데이터
    dates = ["4.23", "4.24", "4.25", "4.26", "4.27", "4.28", "4.29", "4.30", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6"]
    sales_counts = [550, 600, 550, 580, 800, 500, 350, 7000, 4000, 1200, 800, 900, 700, 300]
    avg_prices = [133766.7, 132806.3, 135828.1, 138449.5, 138742.9, 139169.6, 138562.9, 88084.4, 87604.5, 88390, 88217.8, 86910.2, 83283.2, 80962.4]
    # 예측 데이터 (5.7~5.10)
    pred_dates = ["5.7", "5.8", "5.9", "5.10"]
    pred_sales = [5400, 950, 670, 740]
    pred_avg_prices = [68950, 72046, 74206, 73154.6]
    # 호출
    plot_sales_and_price(
        dates, sales_counts, avg_prices,
        pred_dates, pred_sales, pred_avg_prices,
        color_bar='#2196F3', color_bar_pred='#FF5722',
        color_line='#00C853', color_line_pred='#FFC107',
        line_style_actual='-', line_style_pred='--'
    )
