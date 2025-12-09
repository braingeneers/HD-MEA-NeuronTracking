import braingeneers.utils.smart_open_braingeneers as smart_open
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import pandas as pd 


def read_train(qm_path):
    with smart_open.open(qm_path, 'rb') as f:
        with zipfile.ZipFile(f, 'r') as f_zip:
            qm = f_zip.open("qm.npz")
            data = np.load(qm, allow_pickle=True)
            spike_times = data["train"].item()
            fs = data["fs"]
            train = [times / fs for __, times in spike_times.items()]
    return train


def load_curation(qm_path):
    with smart_open.open(qm_path, 'rb') as f:
        with zipfile.ZipFile(f, 'r') as f_zip:
            qm = f_zip.open("qm.npz")
            data = np.load(qm, allow_pickle=True)
            spike_times = data["train"].item()
            fs = data["fs"]
            train = [times / fs for _, times in spike_times.items()]
            if "config" in data:
                config = data["config"].item()
            else:
                config = None
            neuron_data = data["neuron_data"].item()
    return train, neuron_data, config, fs


def load_connectivity(data_path):
    func_pairs = None
    with zipfile.ZipFile(data_path, 'r') as f_zip:
        func_pairs = f_zip.open("func_pairs.npz")
        data = np.load(func_pairs, allow_pickle=True)
        func_pairs = data["func_pairs"].item()
    return func_pairs


def sender_and_receiver_id(func_pairs, latency_threshold=1, ccg_threshold=10):
    '''
    Find senders and receivers from neuron pairs
    Set latency_threhold and ccg_threshold to 0 to let all pairs pass
    Otherwise, filter the pairs using their connection latency and y-value of ccg 
    '''
    senders = set()
    receivers = set()
    senders_only, receivers_only = [], []
    relay = set()
    for pair, ccg_data in func_pairs.items():
        if abs(ccg_data["latency"]) >= latency_threshold and max(ccg_data["ccg"]) >= ccg_threshold:   # filter the pairs using their connection latency 
            p0, p1 = pair[0], pair[1]
            if ccg_data["latency"] > 0:
                receivers.add(p0)
                senders.add(p1)
            else:
                senders.add(p0)
                receivers.add(p1)
    # sort neurons
    for s in list(senders):
        if s not in receivers:
            senders_only.append(s)
        else:
            relay.add(s)

    for r in list(receivers):
        if r not in senders:
            receivers_only.append(r)
        else:
            relay.add(r)
    relay = list(relay)
    return senders_only, receivers_only, relay


def load_umap_df(file_path):
    umap_df_added = pd.read_csv(file_path)
    for i in range(umap_df_added.shape[0]):
        wf = umap_df_added.iloc[i]["waveform"]
        neighbor_waveforms = umap_df_added.iloc[i]["neighbor_waveforms"]
        neighbor_positions = umap_df_added.iloc[i]["neighbor_positions"]
        float_strings = wf.strip('[]').split()     # convert the string respresentation to a list of strings
        # print(float_strings)
        float_neighbor_waveforms = neighbor_waveforms.strip('[]').split()     
        # print(float_neighbor_waveforms) 
        nei_wf = np.array([float(x.rstrip(',')) for x in float_neighbor_waveforms])
        # print(nei_wf)
        float_neighbor_positions = neighbor_positions.strip('[]').split()     
        nei_pos = np.array([float(x.rstrip(',')) for x in float_neighbor_positions])
        # Convert each float string to a float value
        float_wf = np.array([float(value) for value in float_strings])
        # float_neighbor_wf = np.array([float(value) for value in float_neighbor_waveforms])
        # float_neighbor_pos = np.array([float(value) for value in float_neighbor_positions])
        umap_df_added.at[i, "waveform"] = float_wf[:50]   # remove the last 2 values of the electrode position 
        umap_df_added.at[i, "neighbor_waveforms"] = nei_wf
        umap_df_added.at[i, "neighbor_positions"] = nei_pos
    return umap_df_added




## Plotting functions
def plot_inset(axs, temp_pos, templates, nelec=2, ylim_margin=0, pitch=17.5):
    assert len(temp_pos) == len(templates), "Input length must be the same!"
    # find the max template
    if isinstance(templates, list):
        templates = np.asarray(templates)
    amp = np.max(templates, axis=1) - np.min(templates, axis=1)
    max_amp_index = np.argmax(amp)
    position = temp_pos[max_amp_index]
    axs.scatter(position[0], position[1], linewidth=10, alpha=0.2, color='grey')
    axs.text(position[0], position[1], str(position), color="g", fontsize=12)
    # set same scaling to the insets
    ylim_min = min(templates[max_amp_index])
    ylim_max = max(templates[max_amp_index])
    # choose channels that are close to the center channel
    for i in range(len(temp_pos)):
        chn_pos = temp_pos[i]
        if position[0] - nelec * pitch <= chn_pos[0] <= position[0] + nelec * pitch \
                and position[1] - nelec * pitch <= chn_pos[1] <= position[1] + nelec * pitch:
            axin = axs.inset_axes([chn_pos[0]-5, chn_pos[1]-5, 15, 20], transform=axs.transData)
            axin.plot(templates[i], color='k', linewidth=2, alpha=0.7)
            axin.set_ylim([ylim_min - ylim_margin, ylim_max + ylim_margin])
            axin.set_axis_off()
    axs.set_xlim(position[0]-1.5*nelec*pitch, position[0]+1.5*nelec*pitch)
    axs.set_ylim(position[1]-1.5*nelec*pitch, position[1]+1.5*nelec*pitch)
    axs.invert_yaxis()
    return axs


def plot_unit_footprint(qm_path, title="", save_to=None):
    """
    plot footprints for all units in one figure
    """
    _, neuron_data, _, _ = load_curation(qm_path)

    for k, data in neuron_data.items():
        cluster = data["cluster_id"]
        npos = data["neighbor_positions"]
        ntemp = data["neighbor_templates"]

        fig, axs = plt.subplots(figsize=(4, 4))
        axs = plot_inset(axs=axs, temp_pos=npos, templates=ntemp)
        axs.set_title(f"{title} Unit {cluster} ")
        if save_to is not None:
            plt.savefig(f"{save_to}/footprint_{title}_unit_{cluster}.png", dpi=300)
            plt.close()


def plot_functional_map(spike_times, neuron_data, elec_map=None, paired_direction=[], title="", scale=20):
    fig, axs = plt.subplots(figsize=(11, 6))
    axs.set_aspect('equal')
    plt.title(f"{title} Functional Connectivity Map", fontsize=12)
    # draw electrodes
    if elec_map is None or len(elec_map) == 0:
        elec_xy = np.asarray([(x, y) for x in np.arange(0, 3850, 17.5)
                                for y in np.arange(0, 2100, 17.5)])
        axs.scatter(elec_xy[:, 0], elec_xy[:, 1], s=0.2, color='b', alpha=0.3)
    else:
        axs.scatter(elec_map[:, 0], elec_map[:, 1], s=0.2, color='b', alpha=0.3)

    # take the lowest firing rate as a reference
    # ref_fr_min = min([len(spike_times[i]) for i in range(len(spike_times))])
    rec_length = max(times[-1] for times in spike_times)
    chn_pos = np.asarray([data['position'] for _, data in neuron_data.items()])

    if len(paired_direction) == 0:
        for i in range(len(spike_times)):
            axs.scatter(chn_pos[i][0], chn_pos[i][1], s=len(spike_times[i])/rec_length*scale, color='green')
        axs.scatter(None, None, s=scale, color='green', label='Unit 1Hz')

    elif len(paired_direction) > 0:  # unit id is used for paired_direction
        sender = set()
        receiver = set()
        for p in paired_direction:
            sender.add(p[0])
            receiver.add(p[1])
        relay = sender.intersection(receiver)
        paired = sender.union(receiver)
        for p in paired_direction:
            ## p = [i, j, chn_pos[i], chn_pos[j], sttc[i][j], np.mean(lat)]
            color1 = 'gray' if p[0] in relay else 'r'
            color2 = 'gray' if p[1] in relay else 'b'
            axs.scatter(p[2][0], p[2][1], s=len(spike_times[p[0]])/rec_length*scale, color=color1)
            axs.scatter(p[3][0], p[3][1], s=len(spike_times[p[1]])/rec_length*scale, color=color2)
            if 10*p[4] < 1:
                axs.plot([p[2][0], p[3][0]], [p[2][1], p[3][1]], linewidth=1, color='darkgrey', alpha=1)
            elif 10*p[4] > 5:
                axs.plot([p[2][0], p[3][0]], [p[2][1], p[3][1]], linewidth=5, color='darkgrey', alpha=1)
            else:
                axs.plot([p[2][0], p[3][0]], [p[2][1], p[3][1]], linewidth=10*p[4], color='darkgrey', alpha=1)
            # axs.annotate('', xytext=(p[2][0], p[2][1]), xy=((p[2][0] + p[3][0]) / 2, (p[2][1] + p[3][1]) / 2),
            #                 arrowprops=dict(arrowstyle="->", color='darkgrey'), size=8*p[4])
        # for i in range(len(spike_times)):
        #     if i not in paired:
        #         axs.scatter(chn_pos[i][0], chn_pos[i][1], s=len(spike_times[i])/rec_length*scale, color='green')

        axs.scatter(None, None, s=scale, color='r', label='Sender')
        axs.scatter(None, None, s=scale, color='b', label='Receiver')
        axs.scatter(None, None, s=scale, color='gray', label='Relay')

    axs.scatter(None, None, s=scale, color='green', label='Unit')
    axs.scatter(None, None, s=scale, color='green', label='1 Hz')
    axs.scatter(None, None, s=scale*10, color='green', label='10 Hz')
    axs.legend(fontsize=12)
    # axs.legend(loc="upper right", fontsize=12)

    # axs.set_xlim(0, 3850)
    # axs.set_ylim(0, 2100)
    # axs.set_xticks([0, 3850])
    # axs.set_yticks([0, 2100])
    axs.xaxis.set_tick_params(labelsize=12)
    axs.yaxis.set_tick_params(labelsize=12)
    axs.set_xlabel(u"\u03bcm", fontsize=16)
    axs.set_ylabel(u"\u03bcm", fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return axs