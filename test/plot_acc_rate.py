import os
import argparse
import re
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

DEF_PLOT_BS=["all", "best", "1", "4", "8", "10", "15"]
DEF_PLOT_LENGTH=["all", "1000", "3000", "5000", "6000"]

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
    ## others
    extra_bs: Optional[List[int]] = field(default_factory=list)

    def get_plot_label(self):
        if "llama" in self.model:
            return self.model.replace("llama", "l") + "@" + self.name
        else:
            return self.model + "@" + self.name

def load_metrics_from_file(file_path: str, update_model_name: bool) -> Tuple[List[MetricsData], List[MetricsData]]:
    metrics=[]
    best_metrics = []
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
        cur_best_metrics = None
        scan_best_throughput = False
        while line:
            line = line.strip()
            if line.startswith("[BeginMetrics]"):
                cur_metrics = MetricsData(name=name)
                line = f.readline()
                continue
            elif line.startswith("[Best Throughput]"):
                scan_best_throughput = True
                line = f.readline()
                continue
            if scan_best_throughput:
                match = re.search(r'\((\d+),(\d+)\)', line)
                if match:
                    cur_best_metrics = MetricsData(name=name)
                    cur_best_metrics.model = model_name
                    cur_best_metrics.input_len = int(match.group(1))
                    cur_best_metrics.output_len = int(match.group(2))
                    cur_best_metrics.ttft = []
                    cur_best_metrics.extra_bs = []
                    cur_best_metrics.tps = []
                    cur_best_metrics.throughput = []
                    best_metrics.append(cur_best_metrics)
                elif line.startswith("[avg]") or line.startswith("[p50]") or line.startswith("[p90]") or line.startswith("[p99]"):
                    parts = line[6:].split(",")
                    for part in parts:
                        kv = part.split(": ")
                        if "ttft" in kv[0]:
                            cur_best_metrics.ttft.append(float(kv[1]))
                        elif "bs" in kv[0]:
                            cur_best_metrics.extra_bs.append(int(kv[1]))
                        elif "tps" in kv[0]:
                            cur_best_metrics.tps.append(float(kv[1]))
                        elif "throughput" in kv[0]:
                            cur_best_metrics.throughput.append(int(kv[1]))
            else:
                if cur_metrics is None:
                    line = f.readline()
                    continue
                if line.startswith("[EndMetrics]"):
                    metrics.append(cur_metrics)
                    cur_metrics = None
                if line.startswith("model"):
                    if model_name is None:
                        model_name = line.split(":")[1].strip()
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
    return (metrics, best_metrics)

def filter_metrics(metrics, filters: str):
    parts = filters.split(",")
    if "fp8" not in parts and "bf16" not in parts:
        check_fp8 = True
        check_bf16 = True
    else:
        check_fp8 = "fp8"in parts
        check_bf16 = "bf16" in parts
        if check_fp8:
            parts.remove("fp8")
        if check_bf16:
            parts.remove("bf16")
    result = []
    for m in metrics:
        if not (check_fp8 and check_bf16):
            if check_fp8 and "fp8" not in m.model:
                continue
            elif check_bf16 and "fp8" in m.model:
                continue
        keep = False
        for part in parts:
            part = "h2o" if part == "h20" else part
            keep = keep or part in m.name
            if keep:
                break
        if keep:
            result.append(m)
    return result

def find_metrics(metrics, input_len, bs):
    for m in metrics:
        if bs is None:
            if m.input_len == input_len:
                return m
        else:
            if m.batchsize == bs and m.input_len == input_len:
                return m
    return None

def get_plot_style_from_label(label:str) -> Tuple[str, str]:
    ls = "solid"
    ms = "o"
    if "h100" in label:
        ls = "dashed"
        ms = "s"
    elif "h2o" in label:
        ls = "dashdot"
        ms = "v"
    elif "a100" in label:
        ls = "dotted"
        ms = "^"
    elif "a800" in label:
        ls = "-."
        ms = "d"
    elif "l20" in label:
        ls = "--"
        ms = "x"
    elif "l40" in label:
        ls = "-"
        ms = "*"
    elif "a6000" in label:
        ls = ":"
        ms = "."
    return ls, ms

def plot_with_bs(metrics_base, metrics_targets, args):
    plt_bs = args.plot_bs.split(",")
    if "all" in plt_bs:
        plt_bs = DEF_PLOT_BS[2:]
    plt_length = args.plot_length.split(",")
    if "all" in plt_length:
        plt_length = DEF_PLOT_LENGTH[1:]
    print(f"plt_bs: {plt_bs}, plt_length: {plt_length}")
    for i in range(len(metrics_targets)):
        print(f"target[{i}]: {metrics_targets[i][0].get_plot_label()}")

    plt_data = []
    xticks = []
    for i in range(len(metrics_targets)):
        m_target = metrics_targets[i]
        for length in plt_length:
            length = int(length)
            plt_data.append({"label": m_target[0].get_plot_label()+f"#{length}", "x": [], "y": []})
            for bs in plt_bs:
                bs = int(bs)
                xticks.append(bs)
                m_b = find_metrics(metrics_base, length, bs)
                m_t = find_metrics(m_target, length, bs)
                if m_b is None or m_t is None:
                    raise RuntimeError(f"Not found right metrics data, length: {length}, bs: {bs}, base: {m_b}, target: {m_t}")
                plt_data[-1]["x"].append(bs)
                plt_data[-1]["y"].append((m_t.throughput[0] + m_t.throughput[1]) / (m_b.throughput[0] + m_b.throughput[1]))
    print(f"plting data: {plt_data}")
    for data in plt_data:
        ls, ms = get_plot_style_from_label(data["label"])
        plt.plot(data["x"], data["y"], linestyle=ls, marker=ms, label=data["label"])
        if args.plot_ann:
            for i, y in enumerate(data["y"]):
                plt.annotate(f"{data['y'][i]:0.2f}", xy=(data["x"][i], data["y"][i]), xytext=(data["x"][i], data["y"][i]))
    plt.title(f"Speedup vs: {metrics_base[0].get_plot_label()}")
    plt.xlabel("batch size")
    plt.ylabel("speedup")
    plt.xticks(xticks)
    plt.legend()
    #plt.grid(True)
    if args.output:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()

def plot_with_best(best_metrics_base, best_metrics_targets, args):
    print(f"best_metrics_base[{best_metrics_base[0].get_plot_label()}]: {best_metrics_base[0].input_len}, {best_metrics_base[0].ttft}, {best_metrics_base[0].tps}, {best_metrics_base[0].extra_bs}, {best_metrics_base[0].throughput}")
    for i in range(len(best_metrics_targets)):
        print(f"target[{i}]: {best_metrics_targets[i][0].get_plot_label()}")
    plt_length = args.plot_length.split(",")
    if "all" in plt_length:
        plt_length = DEF_PLOT_LENGTH[1:]
    ## x: length(1000, 3000, 5000), y: rate (p90)
    plt_data = []
    x_ticks = [int(i) for i in plt_length]
    for i in range(len(best_metrics_targets)):
        target = best_metrics_targets[i]
        plt_data.append({"label": target[0].get_plot_label(), "x": [], "y": [], "ann": []})
        for length in plt_length:
            length = int(length)
            mb = find_metrics(best_metrics_base, length, None)
            mt = find_metrics(target, length, None)
            if mb is None or mt is None:
                raise RuntimeError(f"Not found right metrics data, length: {length}, base: {m_b}, target: {m_t}")
            plt_data[-1]["x"].append(length)
            plt_data[-1]["y"].append(mt.throughput[3] / mb.throughput[3])
            plt_data[-1]["ann"].append(f"{plt_data[-1]['y'][-1]:.2f}({mt.extra_bs[3]}/{mb.extra_bs[3]})")
    print(f"plting data: {plt_data}")
    for data in plt_data:
        ls, ms = get_plot_style_from_label(data["label"])
        plt.plot(data["x"], data["y"], linestyle=ls, marker=ms, label=data["label"])
        if args.plot_ann:
            for i, y in enumerate(data["y"]):
                plt.annotate(data["ann"][i], xy=(data["x"][i], data["y"][i]), xytext=(data["x"][i], data["y"][i]))
            #plt.annotate(data["label"], xy=(data["x"][-1], data["y"][-1]), xytext=(data["x"][-1], data["y"][-1]))
    plt.title(f"Speedup vs: {best_metrics_base[0].get_plot_label()}")
    plt.xlabel("Context length")
    plt.ylabel("Speedup")
    plt.xticks(x_ticks)
    plt.legend()
    plt.grid(True)
    if args.output:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()

def main(args: argparse.Namespace):
    if not args.base:
        raise ValueError("The baseline is not set")
    if not args.targets:
        raise ValueError("The targets are not set")
    (metrics_base, best_metrics_base) = load_metrics_from_file(args.base, True)
    targets = args.targets.split(",")
    metrics_targets = []
    best_metrics_targets = []
    filter_parts = None
    if args.plot_filter:
        filter_parts = args.plot_filter.split(",")
        if "bf16" in filter_parts:
            filter_parts.remove("bf16")
        if "fp8" in filter_parts:
            filter_parts.remove("fp8")
    target_paths = []
    for t in targets:
        if os.path.isdir(t):
            tt = []
            for subdir in os.listdir(t):
                subpath = os.path.join(t, subdir)
                if os.path.isdir(subpath):
                    if filter_parts is None or len(filter_parts) == 0:
                        tt.append(subpath)
                    elif subdir in filter_parts:
                        tt.append(subpath)
            for subdir in tt:
                for subfilename in os.listdir(subdir):
                    subpath = os.path.join(subdir, subfilename)
                    if os.path.isfile(subpath) and not os.path.samefile(args.base, subpath):
                        target_paths.append(subpath)
        else:
            if not os.path.samefile(args.base, t):
                target_paths.append(t)
    for t in target_paths:
        (m1, m2) = load_metrics_from_file(t, True)
        if args.plot_filter:
            m1 = filter_metrics(m1, args.plot_filter)
            m2 = filter_metrics(m2, args.plot_filter)
        if len(m1) > 0:
            metrics_targets.append(m1)
        if len(m2) > 0:
            best_metrics_targets.append(m2)
    print(f"baseline: {metrics_base[0].get_plot_label()}")
    if "best" in args.plot_bs:
        plot_with_best(best_metrics_base, best_metrics_targets, args)
    else:
        plot_with_bs(metrics_base, metrics_targets, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the total throughput comparison"
    )
    parser.add_argument("--base", type=str, help="The baseline benchmark file")
    parser.add_argument("--targets", type=str, help="The target benchmark files, separated by comma")
    parser.add_argument("--plot-bs", type=str, default="all", help=f"Which batch size to show, can be {DEF_PLOT_BS}, default is all")
    parser.add_argument("--plot-length", type=str, default="all", help="The context lengths, can be {DEF_PLOT_LENGTH}, default is all")
    parser.add_argument("--plot-ann", action="store_true", help="If set, plot the value for each point")
    parser.add_argument("--plot-filter", type=str, help="If set, Filter out the specific metrics, it can be model type(fp8,bf16) or GPU type(4090,a100,h100,h20,...)")
    parser.add_argument("--output", type=str, help="If set, save the graph into the output file")
    args = parser.parse_args()
    main(args)

