import json

import matplotlib.pyplot as plt


def plot_cdf(all_cdfs, all_top_experts):
    for layer_idx, cdf in enumerate(all_cdfs):
        # add square at points
        top_experts = all_top_experts[layer_idx]
        plt.scatter(0, 0, c="#1f77b4", marker="s")
        for i, expert in enumerate(top_experts):
            plt.scatter(i + 1, cdf[i], c="#1f77b4", marker="s")
            plt.text(
                i + 1,
                cdf[i],
                f"{cdf[i]:.2f}",
                fontsize=9,
                verticalalignment="bottom",
                horizontalalignment="center",
            )
        plt.grid()
        cdf = [0] + cdf
        plt.plot(cdf, label=f"Layer {layer_idx}")
        plt.legend()
        plt.xlabel("Experts")
        plt.ylabel("CDF")
        plt.title(f"CDF of top-k experts for layer {layer_idx}")
        plt.savefig(f"plots/layer_{layer_idx}_cdf.png")


def get_cdf_topk_experts(data):
    all_cdfs = []
    all_top_experts = []
    for layer_idx in data.keys():
        total_hits = sum(data[layer_idx].values())
        normalized_hits_per_expert = {}
        for expert in data[layer_idx].keys():
            normalized_hits_per_expert[expert] = data[layer_idx][expert] / total_hits
        # get top three hit prob
        top_experts = sorted(
            normalized_hits_per_expert, key=normalized_hits_per_expert.get, reverse=True
        )

        print(f"Layer {layer_idx}")
        cdf = []
        for i, expert in enumerate(top_experts):
            cdf.append(
                sum(
                    [
                        normalized_hits_per_expert[expert]
                        for expert in top_experts[: i + 1]
                    ]
                )
            )

        all_cdfs.append(cdf)
        all_top_experts.append(top_experts)

        return all_cdfs, all_top_experts


def find_optimal_expert_config(
    all_cdfs,
    num_offload_per_layer,
    num_layers,
    experts_per_layer,
    min_layer_exps,
    max_layer_exps,
):
    # solve a dp to find the optimal expert configuration
    # with the best cdf
    total_offload = num_offload_per_layer * num_layers
    total_load = experts_per_layer * num_layers - total_offload
    dp = [[float("-inf") for _ in range(total_load + 1)] for _ in range(num_layers)]
    exps = [[float("-inf") for _ in range(total_load + 1)] for _ in range(num_layers)]
    for i in range(num_layers):
        for j in range(
            min_layer_exps * (i + 1), min(total_load, max_layer_exps * (i + 1)) + 1
        ):
            if i == 0:
                dp[i][j] = all_cdfs[i][j]
                exps[i][j] = j
                continue

            for k in range(min_layer_exps, max_layer_exps + 1):
                if dp[i][j] < dp[i - 1][j - k] + all_cdfs[i][k]:
                    dp[i][j] = all_cdfs[i][k] + dp[i - 1][j - k]
                    exps[i][j] = k

    # backtrace to get the optimal expert configuration
    exps_per_layer = [exps[num_layers - 1][total_load]]
    load_sum = exps_per_layer[0]
    for i in range(num_layers - 2, -1, -1):
        exps_layer = exps[i][total_load - load_sum]
        load_sum += exps_layer
        exps_per_layer.append(exps_layer)
    exps_per_layer.reverse()
    assert sum(exps_per_layer) == total_load, f"{sum(exps_per_layer)} != {total_load}"

    return dp, exps_per_layer


if __name__ == "__main__":
    with open("expert_hits.json") as f:
        data = json.load(f)

    all_cdfs, all_top_experts = get_cdf_topk_experts(data)

    # -- params
    min_layer_exps = 3
    max_layer_exps = 5
    experts_per_layer = 8
    num_offload_per_layer = 4
    num_load_per_layer = experts_per_layer - num_offload_per_layer
    num_layers = 32
    total_offload = num_offload_per_layer * num_layers
    total_load = experts_per_layer * num_layers - total_offload

    dp, exps_per_layer = find_optimal_expert_config(
        all_cdfs,
        num_offload_per_layer,
        num_layers,
        experts_per_layer,
        min_layer_exps,
        max_layer_exps,
    )
    print(
        f"Naive: {sum([all_cdfs[i][num_load_per_layer] for i in range(num_layers)]) / 32}"
    )
    print(f"Optimal: {dp[num_layers-1][total_load] / 32}")

    # save the experts to load for each layer in json
    experts_to_offload = {}
    for layer_id, num_experts_per_layer in enumerate(exps_per_layer):
        experts_to_offload[layer_id] = all_top_experts[layer_id][num_experts_per_layer:]
    with open("experts_to_offload.json", "w") as f:
        json.dump(experts_to_offload, f, indent=4)
