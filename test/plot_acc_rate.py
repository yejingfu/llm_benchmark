import os
import argparse
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

DEF_PLOT_BS=["all", "1", "4", "8", "10", "15"]
DEF_PLOT_LENGTH=["all", "1000", "3000", "5000"]

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
        return self.model + "@" + self.name

def load_metrics_from_file(file_path: str, update_model_name: bool) -> List[MetricsData]:
    metrics=[]
    print(f"Load metrics from {file_path}")
    if not os.path.isfile(file_path):
        raise RuntimeError(f"Invalid metrics file path: {file_path}")
    base_name = os.path.basename(file_path)
    pos1 = base_name.find("_gpu_")
    pos2 = base_name.find("_", pos1+5)
    if pos1 >= 0 and pos2 > pos1:
        name = base_name[pos1+5:pos2]
    else:
        name = base_name
    model_name = None
    if update_model_name:
        pos1 = base_name.find("_model")
        pos2 = base_name.find("_", pos1 + 7)
        if pos2 < 0:
            pos2 = base_name.find(".", pos1 + 7)
        model_name = base_name[pos1+7:pos2]

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
                if model_name is None:
                    cur_metrics.model = line.split(":")[1].strip()
                else:
                    cur_metrics.model = model_name
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
                cur_metrics.throughput.insert(0, cur_metrics.throughput[0] + cur_metrics.throughput[1])
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

def find_metrics(metrics, input_len, bs):
    for m in metrics:
        if m.batchsize == bs and m.input_len == input_len:
            return m
    return None

def main(args: argparse.Namespace):
    if not args.base:
        raise ValueError("The baseline is not set")
    if not args.targets:
        raise ValueError("The targets are not set")
    metrics_base = load_metrics_from_file(args.base, True)
    targets = args.targets.split(",")
    metrics_targets = []
    for t in targets:
        metrics_targets.append(load_metrics_from_file(t, True))
    print(f"baseline: {metrics_base[0].get_plot_label()}")
    for i in range(len(metrics_targets)):
        print(f"target[{i}]: {metrics_targets[i][0].get_plot_label()}")
    plt_bs = args.plot_bs.split(",")
    if "all" in plt_bs:
        plt_bs = DEF_PLOT_BS[1:]
    plt_length = args.plot_length.split(",")
    if "all" in plt_length:
        plt_length = DEF_PLOT_LENGTH[1:]
    print(f"plt_bs: {plt_bs}, plt_length: {plt_length}")

    plt_data = []
    for i in range(len(metrics_targets)):
        m_target = metrics_targets[i]
        for length in plt_length:
            length = int(length)
            plt_data.append({"label": m_target[0].get_plot_label()+f"#{length}", "x": [], "y": []})
            for bs in plt_bs:
                bs = int(bs)
                m_b = find_metrics(metrics_base, length, bs)
                m_t = find_metrics(m_target, length, bs)
                if m_b is None or m_t is None:
                    raise RuntimeError(f"Not found right metrics data, length: {length}, bs: {bs}, base: {m_b}, target: {m_t}")
                plt_data[-1]["x"].append(bs)
                plt_data[-1]["y"].append((m_t.throughput[0] + m_t.throughput[1]) / (m_b.throughput[0] + m_b.throughput[1]))
    print(f"plting data: {plt_data}")
    for data in plt_data:
        plt.plot(data["x"], data["y"], label=data["label"])
        if args.plot_ann:
            for i, y in enumerate(data["y"]):
                plt.annotate(f"{data['y'][i]:0.2f}", xy=(data["x"][i], data["y"][i]), xytext=(data["x"][i], data["y"][i]))
    plt.title(f"Accelate rate to: {metrics_base[0].get_plot_label()}")
    plt.xlabel("batch size")
    plt.ylabel("speedup")
    plt.legend()
    #plt.grid(True)
    if args.output:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the total throughput comparison"
    )
    parser.add_argument("--base", type=str, help="The baseline benchmark file")
    parser.add_argument("--targets", type=str, help="The target benchmark files, separated by comma")
    parser.add_argument("--plot-bs", type=str, default="all", help=f"Which batch size to show, can be {DEF_PLOT_BS}, default is all")
    parser.add_argument("--plot-length", type=str, default="all", help="The context lengths, can be {DEF_PLOT_LENGTH}, default is all")
    parser.add_argument("--plot-ann", action="store_true", help="If set, plot the value for each point")
    parser.add_argument("--output", type=str, help="If set, save the graph into the output file")
    args = parser.parse_args()
    main(args)

