import matplotlib.pyplot as plt
import sklearn.metrics

import networkx as nx
import numpy as np
import pandas as pd

fontsize=16
minor_size=14

def get_pos(Gp):
    pos = {}
    for node in Gp.nodes():
        r, phi, z = Gp.nodes[node]['pos'][:3]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        pos[node] = np.array([x, y])
    return pos

def plot_nx_with_edge_cmaps(G, weight_name='predict', weight_range=(0, 1),
                            alpha=1.0, ax=None,
                            cmaps=plt.get_cmap('Greys'), threshold=0.):

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    pos = get_pos(G)
    res = [(edge, G.edges[edge][weight_name]) for edge in G.edges() if G.edges[edge][weight_name] > threshold]
    edges, weights = zip(*dict(res).items())

    vmin, vmax = weight_range

    nx.draw(G, pos, node_color='#A0CBE2', edge_color=weights, edge_cmap=cmaps,
            edgelist=edges, width=0.5, with_labels=False,
            node_size=1, edge_vmin=vmin, edge_vmax=vmax,
            ax=ax, arrows=False, alpha=alpha
           )

def plot_metrics(odd, tdd, odd_th=0.5, tdd_th=0.5, outname='roc_graph_nets.eps',
                off_interactive=False, alternative=True):
    if off_interactive:
        plt.ioff()

    y_pred, y_true = (odd > odd_th), (tdd > tdd_th)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, odd)

    if alternative:
        results = []
        labels = ['Accuracy:           ', 'Precision (purity): ', 'Recall (efficiency):']
        thresholds = [0.1, 0.5, 0.8]

        for threshold in thresholds:
            y_p, y_t = (odd > threshold), (tdd > threshold)
            accuracy  = sklearn.metrics.accuracy_score(y_t, y_p)
            precision = sklearn.metrics.precision_score(y_t, y_p)
            recall    = sklearn.metrics.recall_score(y_t, y_p)
            results.append((accuracy, precision, recall))
        
        print("GNN threshold:{:11.2f} {:7.2f} {:7.2f}".format(*thresholds))
        for idx,lab in enumerate(labels):
            print("{} {:6.4f} {:6.4f} {:6.4f}".format(lab, *[x[idx] for x in results]))

    else:
        accuracy  = sklearn.metrics.accuracy_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall    = sklearn.metrics.recall_score(y_true, y_pred)
        print('Accuracy:            %.6f' % accuracy)
        print('Precision (purity):  %.6f' % precision)
        print('Recall (efficiency): %.6f' % recall)

    auc = sklearn.metrics.auc(fpr, tpr)
    print("AUC: %.4f" % auc)
    y_p_5 = odd > 0.5
    print("Fake rejection at 0.5: {:.6f}".format(1-y_true[y_p_5 & ~y_true].shape[0]/y_true[~y_true].shape[0]))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()
    ax0, ax1, ax2, ax3 = axs

    # Plot the model outputs
    # binning=dict(bins=50, range=(0,1), histtype='step', log=True)
    binning=dict(bins=50, histtype='step', log=True)
    ax0.hist(odd[y_true==False], lw=2, label='fake', **binning)
    ax0.hist(odd[y_true], lw=2, label='true', **binning)
    ax0.set_xlabel('Model output', fontsize=fontsize)
    ax0.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax0.legend(loc=0, fontsize=fontsize)
    ax0.set_title('ROC curve, AUC = %.4f' % auc, fontsize=fontsize)

    # Plot the ROC curve
    ax1.plot(fpr, tpr, lw=2)
    ax1.plot([0, 1], [0, 1], '--', lw=2)
    ax1.set_xlabel('False positive rate', fontsize=fontsize)
    ax1.set_ylabel('True positive rate', fontsize=fontsize)
    ax1.set_title('ROC curve, AUC = %.4f' % auc, fontsize=fontsize)
    ax1.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)


    p, r, t = sklearn.metrics.precision_recall_curve(y_true, odd)
    ax2.plot(t, p[:-1], label='purity', lw=2)
    ax2.plot(t, r[:-1], label='efficiency', lw=2)
    ax2.set_xlabel('Cut on model score', fontsize=fontsize)
    ax2.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax2.legend(fontsize=fontsize, loc='upper right')

    ax3.plot(p, r, lw=2)
    ax3.set_xlabel('Purity', fontsize=fontsize)
    ax3.set_ylabel('Efficiency', fontsize=fontsize)
    ax3.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    plt.savefig(outname)
    if off_interactive:
        plt.close(fig)

def np_to_nx(array):
    G = nx.Graph()

    node_features = ['r', 'phi', 'z']
    feature_scales = [1000, np.pi, 1000]

    df = pd.DataFrame(array['x']*feature_scales, columns=node_features)
    node_info = [
        (i, dict(pos=np.array(row), hit_id=array['I'][i])) for i,row in df.iterrows()
    ]
    G.add_nodes_from(node_info)

    receivers = array['receivers']
    senders = array['senders']
    score = array['score']
    truth = array['truth']
    edge_info = [
        (i, j, dict(weight=k, solution=l)) for i,j,k,l in zip(senders, receivers, score, truth)
    ]
    G.add_edges_from(edge_info)
    return G