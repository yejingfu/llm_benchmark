import os
import argparse
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

DEF_PLOT_PERCENT=["none", "avg", "p50", "p90", "p99"]
DEF_DOLLAR_OF_GPU={
    "4090": 0.19,
    "a100": 0.99,
    "h100": 1.86,
    "h2o": 0.57
}
DEF_MARKERS = {
    "4090": ["o", "*", "+"],
    "a100": ["s", "p", "D"],
    "h100": ["^", "v", "<", ">"],
    "h2o": ["|", "_", "+"]
}

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

    def get_label(self):
        return self.name + "#" + self.model.replace("llama", "L")

    def get_dollar_per_mt(self):
        _,_,price = MetricsData.get_gpu_info(self.get_label())
        return price * 1e6 / 3600 / (self.throughput[0] + self.throughput[1])

    def get_max_tps(self, max_ttft):
        max_tps = 0
        for i in range(len(self.raw_ttft)):
            if self.raw_ttft[i] <= max_ttft*1.1 and self.raw_tps[i] > max_tps:
                max_tps = self.raw_tps[i]
        return max_tps

    @classmethod
    def get_gpu_info(cls, label):
        ## parse from name: 1xh100 => 1, "h100", price
        name = label[0:label.find("#")]
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

def get_scatter_marker(label, filtered):
    for x in DEF_MARKERS:
        if x in label:
            for i in DEF_MARKERS[x]:
                if i not in filtered:
                    return i
            return DEF_MARKERS[x][0]
    return "."

def get_scatter_color(bs):
    if bs < 4:
        return "#FF0000" # Red
    elif bs < 8:
        return "#00FF00" # Green
    else:
        return "#0000FF" # Blue

def plot_price_tps(plot_data):
    import matplotlib.pyplot as plt
    keys = list(plot_data.keys())
    fig, axes = plt.subplots(nrows=len(keys), ncols=1)
    for i in range(len(keys)):
        k = keys[i]
        markers = []
        for kk in plot_data[k]:
            marker = get_scatter_marker(kk, markers)
            markers.append(marker)
            c = get_scatter_color(int(kk[kk.rfind("#")+1:]))
            axes[i].scatter([plot_data[k][kk]["price"]], [plot_data[k][kk]["tps"]], s=60, c=c, label=kk, marker=marker)
        axes[i].set_ylabel(f"TPS{k}")
        #axes[i].set_xlabel("$/MTokens")
        #axes[i].grid(True)
    plt.xlabel("$/MTokens")
    plt.legend(loc="best")
    plt.show()

def plot_price_tps_detail(len_to_info):
    import matplotlib.pyplot as plt
    key_len = list(len_to_info.keys())
    fig, axes = plt.subplots(nrows=len(key_len), ncols=1)
    for i in range(len(key_len)):
        k = key_len[i]
        markers = []
        for label in len_to_info[k]:
            x = []
            y = []
            colors = []
            marker = get_scatter_marker(label, markers)
            markers.append(marker)
            for info in len_to_info[k][label]:
                x.append(info["dollar_per_mt"])
                y.append(info["max_tps"])
                colors.append(get_scatter_color(info["bs"]))
            axes[i].scatter(x, y, s=16, label=label, c=colors, marker=marker)
        axes[i].set_ylabel(f"TPS{k}")
        #axes[i].set_xlabel("$/MTokens")
    plt.xlabel("$/MTokens")
    plt.legend(loc="best")
    plt.show()

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
    # name -> (input_len,output_len) -> percentile -> dict(ttft, bs, tps, throughput)
    results = {}
    percentile_labels=["avg", "p50", "p90", "p99"]
    for m in metrics:
        label = m.get_label()
        if label not in results:
            results[label] = {}
        mm = results[label]
        label_length = f"({m.input_len},{m.output_len})"
        if label_length not in mm:
            mm[label_length] = {}
            for i in range(len(percentile_labels)):
                mm[label_length][percentile_labels[i]] = {"ttft":0, "bs":0, "tps":0, "throughput":0}
        mmm = mm[label_length]
        for i in range(len(percentile_labels)):
            if m.ttft[i] < args.max_ttft*1.1 and m.ttft[i] > mmm[percentile_labels[i]]["ttft"]:
                mmm[percentile_labels[i]]["ttft"] = m.ttft[i]
                mmm[percentile_labels[i]]["bs"] = m.batchsize
                mmm[percentile_labels[i]]["tps"] = m.tps[i]
                mmm[percentile_labels[i]]["throughput"] = m.throughput[0]
    #print(f"{results}")
    show_percentile = ["p90", "p99"]
    for k in results:
        m = results[k]
        print(f"\n[{k}]")
        for kk in m:
            print(f"{kk}") ## (input_len,output_len)
            for p in show_percentile:
                mm = m[kk][p]
                print(f"\t[{p}] ttft: {mm['ttft']}, bs: {mm['bs']}, tps: {mm['tps']}, throughput: {int(mm['throughput'])}")
    if args.plot != "none":
        if args.plot in DEF_PLOT_PERCENT:
            print(f"Plot the best of {args.plot}")
            ## (input_len, output_len) => label_bs => dict(dollar_per_mt, tps)
            plot_data = {}
            for label in results:
                m = results[label]
                num, gpu, price = MetricsData.get_gpu_info(label)
                for len_label in m:
                    if args.plot in m[len_label]:
                        mm = m[len_label][args.plot]
                        if len_label not in plot_data:
                            plot_data[len_label] = {}
                        plot_data[len_label][label+f"#{mm['bs']}"] = {"price": price * 1e6 / 3600 / mm['throughput'], "tps": mm['tps']}
            ## ploting
            print(f"\nPlotting: {args.plot}")
            for k in plot_data:
                m = plot_data[k]
                print(k)
                for kk in m:
                    print(f"    [{kk}] Dollar/MTokens: {m[kk]['price']:.3f}, TPS: {m[kk]['tps']}")
            plot_price_tps(plot_data)
        elif args.plot == "detail":
            ## draw more price and tps data on graph
            ## (input_len,output_len) => label => list(dict{bs, dollar_per_mt, max_tps})
            len_to_info = {}
            for m in metrics:
                k = f"({m.input_len},{m.output_len})"
                if k not in len_to_info:
                    len_to_info[k] = {}
                kk = m.get_label()
                if kk not in len_to_info[k]:
                    len_to_info[k][kk] = []
                dpm = m.get_dollar_per_mt()
                max_tps = m.get_max_tps(args.max_ttft)
                len_to_info[k][kk].append({"bs": m.batchsize, "dollar_per_mt": dpm, "max_tps": max_tps})
            print(f"\nPlotting: {args.plot}")
            for len_k in len_to_info:
                print(len_k)
                m = len_to_info[len_k]
                for label in m:
                    print(f"    {label}")
                    for e in m[label]:
                        print(f"      bs:{e['bs']}, dollar_per_mt:{e['dollar_per_mt']}, max_tps:{e['max_tps']}")
            plot_price_tps_detail(len_to_info)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="From the raw benchmark test data, find the best throughput under specific TTFT"
    )
    parser.add_argument("--log-files", type=str, help="The log files, seperated by comma")
    parser.add_argument("--max-ttft", type=float, default=2, help="The maximum ttft, the found throughput should has less ttft than it. Default is 2")
    parser.add_argument("--plot", type=str, default="none", help=f"Draw the graph of specific percentile, can be {DEF_PLOT_PERCENT}, default is none")
    args = parser.parse_args()
    main(args)

