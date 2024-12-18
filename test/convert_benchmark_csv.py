import os
import argparse
import csv
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class MetricsData:
    batchsize: int = field(default=1)
    input_len: int = field(default=0)
    output_len: int = field(default=0)
    ## Percentile: AVG,P50,P90,P99
    latency: List[float] = field(default_factory=list) ## end-to-end
    ttft: List[float] = field(default_factory=list)
    tpot: List[float] = field(default_factory=list)
    tps: List[float] = field(default_factory=list)
    ## in, out, total
    throughput: List[float] = field(default_factory=list)
    rps: float = field(default=0.0)

def load_metrics_from_file(file_path: str, args) -> List[MetricsData]:
    metrics=[]
    print(f"Load metrics from {file_path}")
    if not os.path.isfile(file_path):
        raise RuntimeError(f"Invalid metrics file path: {file_path}")
    with open(file_path, "r") as f:
        line = f.readline()
        cur_metrics = None
        while line:
            line = line.strip()
            if line.startswith("[BeginMetrics]"):
                cur_metrics = MetricsData()
            else:
                if cur_metrics is None:
                    line = f.readline()
                    continue
            if line.startswith("[EndMetrics]"):
                metrics.append(cur_metrics)
                cur_metrics = None
            if line.startswith("model"):
                pass
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
                cur_metrics.throughput.append(cur_metrics.throughput[0] + cur_metrics.throughput[1])
            elif line.startswith("rps"):
                cur_metrics.rps = float(line.split(":")[1].strip())
            line = f.readline()
    return metrics

def main(args: argparse.Namespace):
    if not args.log_files:
        raise ValueError("Invalid log file paths")
    log_files = args.log_files.split(",")
    for path in log_files:
        metrics = load_metrics_from_file(path, args)
        csv_data=[["ctx-length", "bs", "e2e-latency", "ttft", "tpot", "tps", "throughput", "rps"]]
        percent = args.percent.lower()
        for m in metrics:
            e2e = []
            ttft = []
            tpot = []
            tps = []
            if "avg" in percent:
                e2e.append(m.latency[0])
                ttft.append(m.ttft[0])
                tpot.append(m.tpot[0])
                tps.append(m.tps[0])
            if "p50" in percent:
                e2e.append(m.latency[1])
                ttft.append(m.ttft[1])
                tpot.append(m.tpot[1])
                tps.append(m.tps[1])
            if "p90" in percent:
                e2e.append(m.latency[2])
                ttft.append(m.ttft[2])
                tpot.append(m.tpot[2])
                tps.append(m.tps[2])
            if "p99" in percent:
                e2e.append(m.latency[3])
                ttft.append(m.ttft[3])
                tpot.append(m.tpot[3])
                tps.append(m.tps[3])
            s_e2e = ",".join(f"{x:.2f}" for x in e2e)
            s_ttft = ",".join(f"{x:.2f}" for x in ttft)
            s_tpot = ",".join(f"{x:.3f}" for x in tpot)
            s_tps = ",".join(f"{x:.1f}" for x in tps)
            csv_data.append([f"{m.input_len,m.output_len}", f"{m.batchsize}", s_e2e, s_ttft, s_tpot, s_tps, f"{m.throughput[2]:0.1f}", f"{m.rps:.2f}"])
        with open(f"{path}.csv", 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in csv_data:
                writer.writerow(row)
            print(f"The csv benchmark is save to {path}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the llm benchmark data to csv and output to files"
    )
    parser.add_argument("--log-files", type=str, help="The log files, seperated by comma")
    parser.add_argument("--percent", type=str, default="p90,p99", help="Which percentile to convert, default is p90,p99")
    args = parser.parse_args()
    main(args)

