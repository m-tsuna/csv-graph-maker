"""グラフ作成プログラム（課題3）

data/課題3.csv を読み込み、次の3つのグラフを output フォルダに PNG 画像として保存します。
  1. 円グラフ      ：所属ごとの参加者数の割合
  2. 棒グラフ      ：所属ごとのスコアの平均・最高点・最低点
  3. ヒストグラム  ：全参加者のスコアの分布

実行方法：ターミナルで  python main.py
"""

import sys
from pathlib import Path

import matplotlib

# グラフを画面に表示せず、画像ファイルとして保存するだけの設定（pyplot を読み込む前に書く）
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------
# 設定（ここを変えると、ファイルの場所や見た目を変えられます）
# ---------------------------------------------------------------

# このプログラム（main.py）が置いてあるフォルダ
BASE_DIR = Path(__file__).resolve().parent

# 入力するCSVファイルと、画像の保存先フォルダ
CSV_PATH = BASE_DIR / "data" / "課題3.csv"
OUTPUT_DIR = BASE_DIR / "output"

# CSVに必ず入っていてほしい列の名前
REQUIRED_COLUMNS = ["名前", "所属", "スコア"]

# ヒストグラムの区間の幅（5点刻み）
BIN_WIDTH = 5

# グラフの色。色の見え方に個人差があっても区別しやすい組み合わせ（青・オレンジ・緑・黄…の順）
COLORS = [
    "#2a78d6",  # 青
    "#eb6834",  # オレンジ
    "#1baf7a",  # 緑
    "#eda100",  # 黄
    "#e87ba4",  # ピンク
    "#008300",  # 濃い緑
    "#4a3aa7",  # 紫
    "#e34948",  # 赤
]

# 文字や目盛り線の色
TEXT_COLOR = "#0b0b0b"
SUB_TEXT_COLOR = "#52514e"
GRID_COLOR = "#dddddd"

# 日本語を表示できるフォントの候補（上から順に探して、最初に見つかったものを使う）
JAPANESE_FONTS = [
    "Hiragino Sans",  # Mac
    "Hiragino Kaku Gothic ProN",  # Mac
    "YuGothic",  # Mac
    "Yu Gothic",  # Windows
    "Meiryo",  # Windows
    "MS Gothic",  # Windows
    "Noto Sans CJK JP",  # Linux
    "IPAexGothic",  # Linux
]


def exit_with_error(message):
    """エラーメッセージを日本語で表示して、プログラムを終了する。"""
    print(f"【エラー】{message}")
    # 1 は「異常終了」を表す番号（正常終了は 0）
    sys.exit(1)


def setup_japanese_font():
    """グラフの日本語が文字化けしないように、日本語フォントを設定する。"""
    # このパソコンに入っているフォントの名前を集める
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}

    # 候補を上から順に調べ、入っているものがあればそれを使う
    for font_name in JAPANESE_FONTS:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            return

    # 1つも見つからなかった場合は、注意だけ表示して続行する
    print("【注意】日本語フォントが見つかりませんでした。グラフの日本語が文字化けする可能性があります。")


def load_data(csv_path):
    """CSVファイルを読み込み、中身に問題がないか確かめてから表（DataFrame）として返す。（F1・F6）"""
    # ファイルが無い場合は、どこを探したかを伝えて終了する
    if not csv_path.exists():
        exit_with_error(
            f"CSVファイルが見つかりません：{csv_path}\n"
            "　「data」フォルダの中に「課題3.csv」を置いてから、もう一度実行してください。"
        )

    # まず UTF-8 で読み込み、だめなら Shift_JIS（cp932）でも試す
    df = None
    for encoding in ["utf-8-sig", "cp932"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            # この文字コードでは読めなかったので、次の文字コードを試す
            continue
        except pd.errors.EmptyDataError:
            exit_with_error(f"CSVファイルの中身が空です：{csv_path}")
        except pd.errors.ParserError:
            exit_with_error(f"CSVファイルの形式が正しくないため、読み込めませんでした：{csv_path}")

    if df is None:
        exit_with_error(
            f"CSVファイルの文字コードを判別できませんでした：{csv_path}\n"
            "　UTF-8 または Shift_JIS で保存し直してください。"
        )

    # 必要な列がそろっているかを確かめる
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        exit_with_error(
            f"CSVファイルに必要な列がありません：{'、'.join(missing_columns)}\n"
            f"　必要な列：{'、'.join(REQUIRED_COLUMNS)}\n"
            f"　実際の列：{'、'.join(str(column) for column in df.columns)}"
        )

    # データが1行も無い場合はグラフを作れない
    if len(df) == 0:
        exit_with_error("CSVファイルにデータが1行もありません（見出しだけになっています）。")

    # 所属が空欄の行があると集計から漏れてしまうので、先に知らせる
    if df["所属"].isna().any():
        exit_with_error("「所属」列に空欄があります。CSVファイルを確認してください。")

    # スコアが数字として読めるかを確かめる（「八十点」のような文字や空欄があるとエラーにする）
    scores = pd.to_numeric(df["スコア"], errors="coerce")
    if scores.isna().any():
        exit_with_error("「スコア」列に、数字ではない値または空欄があります。CSVファイルを確認してください。")
    df["スコア"] = scores

    return df


def summarize_by_department(df):
    """所属ごとに、人数・割合・スコアの平均・最高点・最低点を計算する。（F2）"""
    # 所属ごとにグループ分けして集計する（sort=False で、CSVに出てきた順番を保つ）
    summary = df.groupby("所属", sort=False)["スコア"].agg(
        人数="count", 平均="mean", 最高点="max", 最低点="min"
    )

    # 全体に対する割合（％）を計算する
    summary["割合"] = summary["人数"] / summary["人数"].sum() * 100

    # 人数の多い順に並べ替える（同じ人数のときは、CSVに出てきた順のまま）
    summary = summary.sort_values("人数", ascending=False, kind="stable")

    # 列を見やすい順番に並べて返す
    return summary[["人数", "割合", "平均", "最高点", "最低点"]]


def summarize_distribution(scores):
    """スコアを5点刻みの区間に分け、区間の境目と、各区間の人数を計算する。（F2・ヒストグラム用）"""
    # 一番低い点・高い点から、区間の始まりと終わりを決める（例：70〜95点 → 70〜100）
    start = int(np.floor(scores.min() / BIN_WIDTH) * BIN_WIDTH)
    end = int(np.floor(scores.max() / BIN_WIDTH) * BIN_WIDTH) + BIN_WIDTH

    # 区間の境目のリストを作る（例：70, 75, 80, 85, 90, 95, 100）
    bin_edges = np.arange(start, end + BIN_WIDTH, BIN_WIDTH)

    # 各区間の人数を数える。
    # 「70点以上75点未満」のように左の値を含み右の値を含まない（最後の区間だけ右端も含む）
    counts, _ = np.histogram(scores, bins=bin_edges)

    return bin_edges, counts


def print_summary(summary, bin_edges, counts, overall_mean):
    """グラフのもとになった数字をターミナルに表示する。（F4）"""
    print("=" * 56)
    print("■ 所属ごとの集計（円グラフ・棒グラフのもとになる数字）")
    print("=" * 56)
    for department, row in summary.iterrows():
        print(
            f"{department}：人数 {int(row['人数'])}人 ／ 割合 {row['割合']:.1f}% ／ "
            f"平均 {row['平均']:.1f} ／ 最高点 {row['最高点']:g} ／ 最低点 {row['最低点']:g}"
        )

    print()
    print("=" * 56)
    print("■ スコアの分布（ヒストグラムのもとになる数字）")
    print("=" * 56)
    last_index = len(counts) - 1
    for i, count in enumerate(counts):
        # 最後の区間だけ右端の点数も含むので「以下」と表示する
        ending = "以下" if i == last_index else "未満"
        print(f"{bin_edges[i]}点以上 {bin_edges[i + 1]}点{ending}：{count}人")
    print(f"合計：{counts.sum()}人")

    print()
    print(f"全体の平均：{overall_mean:.1f} 点")
    print()


def get_colors(count):
    """必要な数だけ色を取り出す。色の数より多い場合は、先頭に戻って繰り返し使う。"""
    return [COLORS[i % len(COLORS)] for i in range(count)]


def style_axes(ax):
    """棒グラフとヒストグラムに共通の見た目（枠線・目盛り線）を整える。"""
    # 上と右の枠線を消して、すっきりさせる
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(SUB_TEXT_COLOR)

    # 横方向の目盛り線を薄く引き、棒の後ろに回す
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SUB_TEXT_COLOR, labelsize=11)


def save_figure(fig, filename):
    """グラフを output フォルダに PNG 画像として保存する。（F5）"""
    # output フォルダが無ければ作る（すでにあっても問題ない）
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 同じ名前のファイルがあれば上書き保存される
    save_path = OUTPUT_DIR / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    # 使い終わったグラフを閉じて、メモリを空ける
    plt.close(fig)
    print(f"保存しました：{save_path}")


def create_pie_chart(summary):
    """円グラフ：所属ごとの参加者数の割合を描いて保存する。（F3-1）"""
    # 各セクションのラベル「所属名（人数）」を作る
    labels = [f"{department}（{int(count)}人）" for department, count in summary["人数"].items()]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        summary["人数"],
        labels=labels,
        colors=get_colors(len(summary)),
        autopct="%.1f%%",  # パーセンテージを小数第1位まで表示
        startangle=90,  # 真上から始める
        counterclock=False,  # 時計回りに並べる
        wedgeprops={"edgecolor": "white", "linewidth": 2},  # セクションの間に白い線を入れる
        textprops={"color": TEXT_COLOR, "fontsize": 13},
    )
    ax.set_title("所属ごとの参加者数の割合", fontsize=16, color=TEXT_COLOR, pad=16)

    save_figure(fig, "pie_chart.png")


def create_bar_chart(summary):
    """棒グラフ：所属ごとのスコアの平均・最高点・最低点を描いて保存する。（F3-2）"""
    # 並べる棒の種類（summary の列名と同じ）
    items = ["平均", "最高点", "最低点"]
    colors = get_colors(len(items))

    # 所属の位置（0, 1, 2, …）と、棒1本の幅
    x_positions = np.arange(len(summary))
    bar_width = 0.26

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, item in enumerate(items):
        # 3本の棒が所属の位置を中心に横に並ぶよう、少しずつずらす
        offset = (i - (len(items) - 1) / 2) * bar_width
        bars = ax.bar(
            x_positions + offset,
            summary[item],
            width=bar_width,
            label=item,
            color=colors[i],
            edgecolor="white",
            linewidth=1.5,
        )

        # 棒の上に数値を表示する（平均は小数第1位まで、最高点・最低点はそのまま）
        number_format = "%.1f" if item == "平均" else "%g"
        ax.bar_label(bars, fmt=number_format, padding=3, fontsize=10, color=TEXT_COLOR)

    ax.set_title("所属ごとのスコア（平均・最高点・最低点）", fontsize=16, color=TEXT_COLOR, pad=16)
    ax.set_xlabel("所属", fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel("スコア（点）", fontsize=13, color=TEXT_COLOR)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary.index, fontsize=12)

    # Y軸は0点から始め、上は数値の文字が入るよう少し余裕を持たせる
    ax.set_ylim(0, summary["最高点"].max() * 1.15)

    style_axes(ax)
    # 凡例は、棒や数値と重ならないようにグラフの右外側に置く
    ax.legend(title="凡例", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=11)

    save_figure(fig, "bar_chart.png")


def create_histogram(bin_edges, counts, overall_mean):
    """ヒストグラム：全参加者のスコアの分布を描いて保存する。（F3-3）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 集計済みの人数を、区間の左端にそろえて棒として描く（数え直さないので表示と必ず一致する）
    bars = ax.bar(
        bin_edges[:-1],
        counts,
        width=BIN_WIDTH,
        align="edge",
        color=COLORS[0],
        edgecolor="white",
        linewidth=2,
        label="人数",
    )

    # 棒の上に人数を表示する
    ax.bar_label(bars, labels=[f"{count}人" for count in counts], padding=3, fontsize=11, color=TEXT_COLOR)

    # 全体の平均点の位置に、縦の点線を引く
    ax.axvline(
        overall_mean,
        color=TEXT_COLOR,
        linestyle=":",
        linewidth=2,
        label=f"全体の平均（{overall_mean:.1f}点）",
    )

    ax.set_title("スコアの分布", fontsize=16, color=TEXT_COLOR, pad=16)
    ax.set_xlabel("スコア（点）", fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel("人数（人）", fontsize=13, color=TEXT_COLOR)

    # X軸の目盛りを区間の境目（70, 75, 80, …）に合わせる
    ax.set_xticks(bin_edges)

    # Y軸は整数の目盛りにして、上は凡例と数値の文字が入るよう余裕を持たせる
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, counts.max() * 1.3)

    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    save_figure(fig, "histogram.png")


def main():
    """プログラム全体の流れ：読み込み → 集計 → 数字の表示 → グラフの作成と保存。"""
    setup_japanese_font()

    # 読み込み（F1・F6）
    df = load_data(CSV_PATH)

    # 集計（F2）
    summary = summarize_by_department(df)
    bin_edges, counts = summarize_distribution(df["スコア"])
    overall_mean = df["スコア"].mean()

    # 集計結果の表示（F4）
    print_summary(summary, bin_edges, counts, overall_mean)

    # グラフの作成と保存（F3・F5）
    create_pie_chart(summary)
    create_bar_chart(summary)
    create_histogram(bin_edges, counts, overall_mean)

    print()
    print("3つのグラフを保存しました。")


# このファイルを「python main.py」で直接実行したときだけ、main() を動かす
if __name__ == "__main__":
    main()
