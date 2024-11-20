import os
import argparse
import re
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

DEF_DOLLAR_OF_GPU={
    "4090": 0.19,
    "a100": 0.99,
    "h100": 1.86,
    "h2o": 0.57
}

DEF_MERGE_SAME_GPU=False

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

    @classmethod
    def get_gpu_info(cls, label):
        ## parse from name: 1xh100 => 1, "h100", price
        name = label[label.find("@")+1:]
        num = 1
        gpu = name
        price = 0
        pos = name.find("x")
        if pos > 0:
            num = int(name[0:pos])
            gpu = name[pos+1:]
        for key in DEF_DOLLAR_OF_GPU:
            if gpu.find(key) == 0:
                price = num * DEF_DOLLAR_OF_GPU[key]
                break;
        return num, gpu, price

def load_metrics_from_file(file_path: str, update_model_name: bool) -> List[MetricsData]:
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
        cur_best_metrics = None
        scan_best_throughput = False
        while line:
            line = line.strip()
            if line.startswith("[Best Throughput]"):
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
            line = f.readline()
    return best_metrics

def filter_metrics(metrics, filters: str, input_len: int):
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
        if m.input_len != input_len:
            continue
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
            for key in DEF_DOLLAR_OF_GPU:
                if key in m.name:
                    result.append(m)
                    break
    return result

def plot_best_solution(metrics, args):
    plt_data = []
    for m in metrics:
        m = m[0]
        data = {"label": m.get_plot_label(), "gpu": "", "x": 0, "y": 0}
        num, gpu, price = MetricsData.get_gpu_info(m.get_plot_label())
        data["gpu"] = gpu
        data["x"] = price * 1e6 / 3600 / m.throughput[3]
        data["y"] = m.throughput[3] / num
        found = False
        if DEF_MERGE_SAME_GPU:
            for i in range(len(plt_data)):
                if plt_data[i]["gpu"] == gpu:
                    found = True
                    if plt_data[i]["y"] / plt_data[i]["x"] < data["y"] / data["x"]:
                        del plt_data[i]
                        plt_data.append(data)
                    break
        if not found:
            plt_data.append(data)
    #print(plt_data)
    for data in plt_data:
        plt.scatter(data["x"], data["y"], marker="o", s=60, label=data["label"])
        if args.plot_ann:
            plt.annotate(data["label"], xy=(data["x"], data["y"]), xytext=(data["x"]*1.01, data["y"]+20), arrowprops=dict(arrowstyle='-', color='grey', lw=1))
    plt.xlabel("$/MTokens")
    plt.ylabel("Tokens/GPU/sec")
    plt.title("speed vs cost")
    if not args.plot_ann:
        plt.legend(loc="best")
    if args.output:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()

def main(args: argparse.Namespace):
    if not args.log_files:
        raise ValueError("The log files are not set")
    dirs = args.log_files.split(",")
    filter_parts = None 
    if args.plot_filter:
        filter_parts = args.plot_filter.split(",")
        if "bf16" in filter_parts:
            filter_parts.remove("bf16")
        if "fp8" in filter_parts:
            filter_parts.remove("fp8")
    if filter_parts is None or len(filter_parts) == 0:
        filter_parts = list(DEF_DOLLAR_OF_GPU.keys())
        if not args.plot_filter:
            args.plot_filter = ",".join(list(DEF_DOLLAR_OF_GPU.keys()))
        else:
            args.plot_filter = args.plot_filter + "," + ",".join(list(DEF_DOLLAR_OF_GPU.keys()))
    paths = []
    for t in dirs:
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
                    if os.path.isfile(subpath):
                        paths.append(subpath)
        else:
            paths.append(t)
    metrics = []
    for t in paths:
        m = load_metrics_from_file(t, True)
        if args.plot_filter:
            m = filter_metrics(m, args.plot_filter, args.plot_length)
        if len(m) > 0:
            metrics.append(m)
    for m in metrics:
        print(f"metric: {m[0].get_plot_label()}, length: {m[0].input_len}")
    plot_best_solution(metrics, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the total throughput comparison"
    )
    parser.add_argument("--log-files", type=str, help="The benchmark files, can be file or directory, separated by comma")
    parser.add_argument("--plot-filter", type=str, help="If set, Filter out the specific metrics, it can be model type(fp8,bf16) or GPU type(4090,a100,h100,h20,...)")
    parser.add_argument("--plot-ann", action="store_true", help="If set, plot the value for each point")
    parser.add_argument("--plot-length", type=int, default=6000, help="Only plot the metrics with the specific length, default is 6000")
    parser.add_argument("--output", type=str, help="If set, save the graph into the output file")
    args = parser.parse_args()
    main(args)

