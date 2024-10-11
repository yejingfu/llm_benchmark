import os
import argparse
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

DEF_PLOT_KIND=["all", "ttft", "tpot", "tps", "throughput"]
DEF_PLOT_BS=["all", "1", "2", "4", "8", "10", "15", "20"]
DEF_PLOT_PERCENT=["all", "avg", "p50", "p90", "p99"]

@dataclass
class MetricsData:
    name: str = field(default="")
    model: str = field(default="")
    batchsize: int = field(default=1)
    input_len: int = field(default=0)
    output_len: int = field(default=0)
    ## Percentile: AVG,P50,P90,P99
    latency: List[float] = field(default_factory=list) ## end-to-end
    ttft: List[float] = field(default_factory=list)
    tpot: List[float] = field(default_factory=list)
    tps: List[float] = field(default_factory=list)
    ## summary
    throughput: List[float] = field(default_factory=list)
    ## raw data
    raw_ttft: Optional[List[float]] = field(default_factory=list)
    raw_tpot: Optional[List[float]] = field(default_factory=list)
    raw_tps: Optional[List[float]] = field(default_factory=list)
    
    def get_plot_label(self):
        return self.name + "#" + self.model + "#(" + str(self.input_len) + "," + str(self.output_len) + ")"

def plot_bars_percentile(ax, labels, dataset, percent):
    block_width = 4
    bar_pad = 1
    bar_width = 0.16
    num_bars = len(labels)
    num_blocks = len(percent)
    xticks = [x+"(" for x in percent]
    xticks_pos = None
    for i in range(num_bars):
        pos=[]
        for j in range(num_blocks):
            pos.append(bar_width * i + j * block_width)
            #xticks.append(f"{percent[j]}({dataset[i][j]:.2f})")
            xticks[j] += f"{dataset[i][j]:.2f},"
        ax.bar(pos, dataset[i], width=bar_width, label=labels[i])
        if xticks_pos is None:
            xticks_pos = pos
    xticks = [x[:-1]+")" for x in xticks]
    ax.set_xticks(pos, xticks)

def plot_bars_throughput(ax, labels, dataset):
    block_width = 3
    bar_pad = 1
    bar_width = 0.16
    num_bars = len(labels)
    num_blocks = 2
    xticks = ["in(", "out("]
    xticks_pos = None
    for i in range(num_bars):
        pos=[]
        for j in range(num_blocks):
            pos.append(bar_width * i + j * block_width)
            xticks[j] += f"{int(dataset[i][j])},"
        ax.bar(pos, dataset[i], width=bar_width, label=labels[i])
        if xticks_pos is None:
            xticks_pos = pos
    xticks = [x[:-1]+")" for x in xticks]
    ax.set_xticks(pos, xticks)

def plot_group_bars(kind, data, bss, percent):
    # every row for each batchsize
    print(f"ploting {kind}, bs: {bss}, percentile: {percent}")
    fig, axes = plt.subplots(nrows=len(bss), ncols=1)
    for i in range(len(bss)):
        dataset = []
        labels = []
        for key in data:
            labels.append(key)
            if bss[i] in data[key]:
                dataset.append(data[key][bss[i]])
            elif kind == "throughput":
                dataset.append([0, 0])
            else:
                dataset.append([0, 0, 0, 0])
        #print(f"labels: {labels}, dataset: {dataset}")
        if kind == "throughput":
            plot_bars_throughput(axes[i], labels, dataset)
        else:
            plot_bars_percentile(axes[i], labels, dataset, percent)
        #axes[i].set_title(f"{kind}, bs={bss[i]}")
        axes[i].set_xlabel(f"{kind}, bs={bss[i]}")
        #axes[i].set_ylabel(kind)
        axes[i].grid(True)
        #axes[i].legend(loc="best")
    fig.subplots_adjust(hspace=0.3)
    plt.legend(loc="best")
    #plt.tight_layout()
    plt.show()

def plot_raw_ttft_tps(metrics):
    for i in range(len(metrics)):
        m = metrics[i]
        if m.raw_ttft is not None and m.raw_tps is not None:
            if m.batchsize in [1, 25]:
                plt.scatter(m.raw_ttft, m.raw_tps, s=1, label=f"{m.get_plot_label()}#{m.batchsize}")
    plt.xlabel("TTFT")
    plt.ylabel("TPS")
    plt.legend()
    plt.show()

def load_metrics_from_file(file_path: str) -> List[MetricsData]:
    metrics=[]
    print(f"Load metrics from {file_path}")
    if not os.path.isfile(file_path):
        raise RuntimeError(f"Invalid metrics file path: {file_path}")
    name = os.path.basename(file_path)
    pos1 = name.find("_gpu_")
    pos2 = name.find("_", pos1+5)
    if pos1 >= 0 and pos2 > pos1:
        name = name[pos1+5:pos2]
    with open(file_path, "r") as f:
        line = f.readline()
        cur_metrics = None
        while line:
            line = line.strip()
            if line.startswith("[BeginMetrics]"):
                cur_metrics = MetricsData(name=name)
            else:
                if cur_metrics is None:
                    line = f.readline()
                    continue
            if line.startswith("[EndMetrics]"):
                metrics.append(cur_metrics)
                cur_metrics = None
            if line.startswith("model"):
                cur_metrics.model = line.split(":")[1].strip()
            elif line.startswith("sequence-length"):
                line = line.split(":")[1].strip()
                lens = line.split(",")
                cur_metrics.input_len = int(lens[0].strip())
                cur_metrics.output_len = int(lens[1].strip())
            elif line.startswith("batch-size"):
                cur_metrics.batchsize = int(line.split(":")[1].strip())
            elif line.startswith("e2e-latency"):
                cur_metrics.latency = eval("[" + line.split(":")[1].strip() + "]")
            elif line.startswith("ttft"):
                cur_metrics.ttft = eval("[" + line.split(":")[1].strip() + "]")
            elif line.startswith("tpot"):
                cur_metrics.tpot = eval("[" + line.split(":")[1].strip() + "]")
            elif line.startswith("tps"):
                cur_metrics.tps = eval("[" + line.split(":")[1].strip() + "]")
            elif line.startswith("throughput"):
                line = line.split(":")[1].strip()
                thp = line.split(",")
                cur_metrics.throughput = [float(thp[0].strip()), float(thp[1].strip())]
            elif line.startswith("rps"):
                cur_metrics.rps = float(line.split(":")[1].strip())
            elif line.startswith("raw-ttft"):
                cur_metrics.raw_ttft = eval(line.split(":")[1].strip())
            elif line.startswith("raw-tpot"):
                cur_metrics.raw_tpot = eval(line.split(":")[1].strip())
            elif line.startswith("raw-tps"):
                cur_metrics.raw_tps = eval(line.split(":")[1].strip())
            line = f.readline()
    return metrics

def main(args: argparse.Namespace):
    if not args.log_files:
        raise ValueError("Invalid log file paths")
    metrics = []
    log_files = args.log_files.split(",")
    for path in log_files:
        tmp = load_metrics_from_file(path)
        for m in tmp:
            metrics.append(m)
    print(f"Total metrics: {len(metrics)}")
    #for i in range(0, len(metrics)):
    #    print(f"Metrics[{i}]: {metrics[i]}")
    plt_kinds = args.plot_kind.split(",")
    if "all" in plt_kinds:
        plt_kinds = DEF_PLOT_KIND[1:]
    plt_bs = args.plot_bs.split(",")
    if "all" in plt_bs:
        plt_bs = DEF_PLOT_BS[1:]
    tmp_plt_percentile = args.plot_precent.split(",")
    if "all" in tmp_plt_percentile:
        tmp_plt_percentile = DEF_PLOT_PERCENT[1:]
    valid_percentile = []
    for p in tmp_plt_percentile:
        if p in DEF_PLOT_PERCENT:
            valid_percentile.append(p)
    # all_data: kind => label => bs
    all_data = {}
    valid_bs = []
    for kind in plt_kinds:
        if kind in DEF_PLOT_KIND and kind != "raw":
            if kind not in all_data:
                all_data[kind] = {}
            for m in metrics:
                label = m.get_plot_label()
                if label not in all_data[kind]:
                    all_data[kind][label] = {}
                data = all_data[kind][label]
                for bs in plt_bs:
                    if int(bs) == m.batchsize:
                        if bs not in valid_bs:
                            valid_bs.append(bs)
                        if kind == "ttft":
                            data[bs] = m.ttft
                        elif kind == "tpot":
                            data[bs] = m.tpot
                        elif kind == "tps":
                            data[bs] = m.tps
                        elif kind == "throughput":
                            data[bs] = m.throughput
    print(f"ploting data: {all_data}")
    for k in all_data:
        plot_group_bars(k, all_data[k], valid_bs, valid_percentile)
    if args.plot_raw:
        plot_raw_ttft_tps(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the benchmark metrics"
    )
    parser.add_argument("--log-files", type=str, help="The log files, seperated by comma")
    parser.add_argument("--plot-kind", type=str, default="all", help=f"Which graph to show, can be {DEF_PLOT_KIND}, default is all")
    parser.add_argument("--plot-bs", type=str, default="all", help=f"Which batch size to show, can be {DEF_PLOT_BS}, default is all")
    parser.add_argument("--plot-precent", type=str, default="all", help=f"Which percentile to show, can be {DEF_PLOT_PERCENT}, default is all")
    parser.add_argument("--plot-raw", action="store_true", help="Draw the raw ttft-tps graph")
    args = parser.parse_args()
    main(args)