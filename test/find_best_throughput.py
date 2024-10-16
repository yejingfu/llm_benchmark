import os
import argparse
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

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
        return self.name + "#" + self.model

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
    # name -> (input_len,output_len) -> dict(ttft, bs, tps, throughput)
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
        print(f"\n[{k}]\n")
        for kk in m:
            print(f"{kk}") ## (input_len,output_len)
            for p in show_percentile:
                mm = m[kk][p]
                print(f"\t[{p}] ttft: {mm['ttft']}, bs: {mm['bs']}, tps: {mm['tps']}, throughput: {int(mm['throughput'])}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="From the raw benchmark test data, find the best throughput under specific TTFT"
    )
    parser.add_argument("--log-files", type=str, help="The log files, seperated by comma")
    parser.add_argument("--max-ttft", type=float, default=2, help="The maximum ttft, the found throughput should has less ttft than it. Default is 2")
    args = parser.parse_args()
    main(args)

