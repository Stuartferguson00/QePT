import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib.cm as cm
from pathlib import Path


def load_pt_results(n_spins_list, m_replicas_list, results_dir, proposals_list, m):
    """
    Load the results from the parallel tempering experiments.
    """
    

    results = {
        'mean_optimal_efforts': [],
        'sem_optimal_efforts': [],
        'mean_optimal_nhops': [],
        'sem_optimal_nhops': [],
        'data_save': []
    } 

    results_path = Path(__file__).resolve().parent
    print("results_path:", results_path)
    for i, n_spins in enumerate(n_spins_list):
        for j, m_replicas in enumerate(m_replicas_list):
            for k, proposals in enumerate(proposals_list):
                
                
                results_dir = results_path / f"results_PT"
                results_dir = results_dir / f"opt_results"

                    
                
                replica_tag = ""
                for proposal in proposals:
                    replica_tag += proposal[0]
                if replica_tag == "l"*len(proposals):
                    res_path = results_dir / f"{str(n_spins).zfill(3)}_{m_replicas}.pkl"
                else:
                    res_path = results_dir / f"{str(n_spins).zfill(3)}_{m_replicas}_{replica_tag}_{m}.pkl"
                                    
                
                try:
                    with open(res_path, 'rb') as f:
                        data = pickle.load(f)
                    print("data['data_save']", data['data_save'])
                    results['mean_optimal_efforts'].append(data['mean_optimal_efforts'])
                    results['sem_optimal_efforts'].append(data['sem_optimal_efforts'])
                    results['mean_optimal_nhops'].append(data['mean_optimal_nhops'])
                    results['sem_optimal_nhops'].append(data['sem_optimal_nhops'])
                    results['data_save'].append(data['data_save'])
                    print("Loaded results from", res_path)
                except FileNotFoundError:
                    for key in results:
                        results[key].append(np.nan)
                    print("No results file found at", res_path)

    for key, value in results.items():
        results[key] = value#np.array(value)
    return results

    
                
    
#return optimal_nhops, sem_optimal_nhops, optimal_efforts, sem_optimal_efforts

def main(results_dir, results_path,n_spins_list,m_replicas_list, m):
    """
    Main function to generate and save the plots.
    """
    # Get PT results
    pt_nhops, pt_sem_nhops, pt_efforts, pt_sem_efforts = load_pt_results(n_spins_list, m_replicas_list, results_dir, m)


  
    fig_2, axs_2 = plt.subplots(1, 3, figsize=(15, 5))

    nan_mask = np.isnan(pt_efforts)
    pt_efforts[nan_mask] = np.inf
    amin = np.argmin(pt_efforts, axis=1)

    # Fit and plot exponential curve for SA n_hops
    def exp_func(x, a, b):
        return a * np.exp(b * x)


    axs_2[0].errorbar(n_spins_list, pt_nhops[np.arange(len(n_spins_list)), amin], yerr=pt_sem_nhops[np.arange(len(n_spins_list)), amin], marker='o', label='PT', elinewidth=0.3, linewidth = 0)
    
    
    
    not_nan_mask = np.invert(nan_mask[:,0])
    # Plot Optimal m_replicas
    axs_2[1].plot(np.array(n_spins_list)[not_nan_mask], np.array(m_replicas_list[amin])[not_nan_mask], marker='o', label='PT', linewidth=0)
    axs_2[1].set_ylabel('Optimal m_replicas')

    # Plot Optimal Effort
    axs_2[2].errorbar(n_spins_list, pt_efforts[np.arange(len(n_spins_list)), amin], yerr=pt_sem_efforts[np.arange(len(n_spins_list)), amin], marker='o', label='PT', elinewidth=0.3, linewidth = 0)
    axs_2[2].set_ylabel('Optimal Effort')
    axs_2[2].set_yscale('log')
    
    fig_2.supxlabel('n_spins')
    fig_2.tight_layout()

    # Plot effort vs m_replicas
    fig_3, axs_3 = plt.subplots(1, 1, figsize=(5, 5))
    cmap_even = plt.get_cmap('cool', len(n_spins_list))
    cmap_odd = plt.get_cmap('Wistia', len(n_spins_list))

    for i, n_spins in enumerate(n_spins_list):
        if n_spins % 2 == 0:
            color = cmap_even(i / len(n_spins_list))
        else:
            color = cmap_odd(i / len(n_spins_list))
        axs_3.errorbar(m_replicas_list, pt_efforts[i, :], yerr=pt_sem_efforts[i, :], marker='o', label=f'n_spins {n_spins}', color=color)

    axs_3.set_xlabel('m_replicas')
    axs_3.set_ylabel('Optimal Effort')
    axs_3.set_title('Optimal Effort vs m_replicas for different n_spins')
    axs_3.set_yscale('log')
    axs_3.legend()
    
    
    fig_2.savefig(results_path / 'fig_optimal_results.png')
    fig_3.savefig(results_path / 'fig_effort_vs_mreplicas.png')
    print("figure saved to ", results_path / 'fig_optimal_results.png')
    print("figure saved to ", results_path / 'fig_effort_vs_mreplicas.png')


def plot_optimal_m_replicas(pt_efforts, pt_sem_efforts):
    #print(pt_nhops, pt_sem_nhops, pt_efforts, pt_sem_efforts)
    plt.figure(figsize=(6, 4))
    cmap = plt.get_cmap('viridis', len(m_replicas_list))
    for i, m_replicas in enumerate(m_replicas_list):
        print("m_replicas_list:", m_replicas_list)
        print("np.where(m_replicas_list == m_replicas)",np.where(m_replicas_list == m_replicas))
        idx_m = np.where(m_replicas_list == m_replicas)[0][0]
        try:
            color = cmap(i / (len(m_replicas_list) - 1))
        except:
            color = "coral"
        plt.errorbar(n_spins_list, pt_efforts[:, idx_m], yerr=pt_sem_efforts[:, idx_m], marker='o', elinewidth=0.3, linewidth=0, color=color, label=f'm={m}')
        
        # Fit and plot exponential curve for each m_replicas
        mask = ~np.isnan(pt_efforts[:, idx_m])
        if np.sum(mask) > 2:
            x = np.array(n_spins_list)[mask]
            y = pt_efforts[:, idx_m][mask]
            # Fit y = a * exp(b * x)
            def exp_func(x, a, b):
                return a * np.exp(b * x)
            try:
                
                popt, _ = curve_fit(exp_func, x, y, p0=(1, 0.1), sigma=pt_sem_efforts[:, idx_m][mask], absolute_sigma=True)
                x_fit = np.linspace(min(x), max(x), 100)
                y_fit = exp_func(x_fit, *popt)
                plt.plot(x_fit, y_fit, '--', color=color, alpha=0.7)
            except Exception:
                pass
        else:
            print("Not enough data points to fit for m_replicas =", m_replicas)
    plt.yscale('log')
    plt.xlabel('n_spins')
    plt.ylabel('Optimal efforts')
    plt.title('Optimal efforts for m_replicas')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path / 'fig_optimal_efforts_mreplicas.png')
    
    
def plot_optimal_effort_n_spins_m_const(pt_efforts, pt_sem_efforts, m_replicas_const, n_spins_list, m_replicas_list, m_quantum_replicas_list):
    #print(pt_nhops, pt_sem_nhops, pt_efforts, pt_sem_efforts)
    plt.figure(figsize=(6, 4))
    
    cmap = plt.get_cmap('viridis', len(m_quantum_replicas_list))
        
    idx_m = np.where(np.isclose(m_replicas_list,m_replicas_const))[0][0]
    for idx_m_q, m_q in enumerate(m_quantum_replicas_list):
        
        #idx_m_q = np.where(np.isclose(m_quantum_replicas_list,m_q))[0][0]
        color = cmap(idx_m_q / (len(m_quantum_replicas_list) - 1))
        if m_q ==0:
            label = 'All classical chains'
        else:
            label = f'{m_q} quantum chains'
        plt.errorbar(n_spins_list, pt_efforts[:, idx_m,idx_m_q], yerr=pt_sem_efforts[:, idx_m,idx_m_q], marker='o', elinewidth=0.3, linewidth=0, color=color, label=label)

        # Fit and plot exponential curve for each m_replicas
        mask = ~np.isnan(pt_efforts[:, idx_m,idx_m_q])
        if np.sum(mask) > 2:
            x = np.array(n_spins_list)[mask]
            y = pt_efforts[:, idx_m][mask]
            # Fit y = a * exp(b * x)
            def exp_func(x, a, b):
                return a * np.exp(b * x)
            try:
                
                popt, _ = curve_fit(exp_func, x, y, p0=(1, 0.1), sigma=pt_sem_efforts[:, idx_m,idx_m_q][mask], absolute_sigma=True)
                x_fit = np.linspace(min(x), max(x), 100)
                y_fit = exp_func(x_fit, *popt)
                plt.plot(x_fit, y_fit, '--', color=color, alpha=0.7)
            except Exception:
                pass
        else:
            print("Not enough data points to fit for m_q =", m_q)
        
    
    
    plt.yscale('log')
    plt.xlabel('n_spins')
    plt.ylabel('Optimal efforts')
    plt.title('Optimal efforts for '+str(m_replicas_const)+" replicas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path / ('fig_optimal_efforts_vs_n_spins_m_'+str(m_q)+"_"+str(m_replicas_const)+'.png'))





def plot_effort_vs_nhops(n_spins_list, m_replicas_list, m_quantum_replicas_list, results, output_dir, label, colormap):
    """
    Plots effort vs n_hops for each n_spins in its own subplot.
    effort and hops labelling is inverted here by accident
    """
    markerslist = ['v', '^', 'x', '+', 'o', 'D', '*', 'X']
    markersizelist = [6,6,6,6,2,6,6,6,6]

    spins_to_plot = n_spins_list

    def process_and_plot(ax, data, label, color, n_spins):
        if isinstance(data, np.ndarray):
            # this is the wrong way around...
            efforts = data[:, 0]
            nhops = data[:, 2]

            print("number of reps (assuming 100 models):", len(efforts)/100)
            unique_efforts = np.unique(efforts)
            mean_nhops = []
            sem_nhops = []

            for effort in unique_efforts:
                hops_at_effort = nhops[efforts == effort]
                mean_nhops.append(np.mean(hops_at_effort))
                sem_nhops.append(np.std(hops_at_effort, ddof=1) / np.sqrt(len(hops_at_effort)) if len(hops_at_effort) > 1 else 0)
                #ax.scatter(np.ones(len(hops_at_effort))*effort, hops_at_effort, marker='x', color=color, s=10, alpha = 0.1)
            ax.errorbar(unique_efforts, mean_nhops, yerr=sem_nhops, label=label, marker="o", linestyle="", markersize=4, color=color)
            
            
            
            # Fit quadratic to the datapoints surrounding the lowest and plot
            if len(unique_efforts) >= 3:
                # Sort by mean_nhops and take lowest 10
                #sorted_indices = np.argsort(mean_nhops)[:5]

                # Find index of minimum mean_nhops
                #min_idx = np.argmin(mean_nhops)
                # Take 5 on each side of the minimum (total up to 11 points)
                # start_idx = max(min_idx - 7, 0)
                # end_idx = min(min_idx + 7 + 1, len(mean_nhops))
                # selected_indices = np.arange(start_idx, end_idx)

                # Get indices of the 0 lowest mean_nhops
                selected_indices = np.argsort(mean_nhops)[:10]
                if 0 in selected_indices:
                    selected_indices = np.delete(selected_indices, np.where(selected_indices == 0)) 
                x_fit = unique_efforts[selected_indices]
                y_fit = np.array(mean_nhops)[selected_indices]
                y_err_fit = np.array(sem_nhops)[selected_indices]

                ax.scatter(x_fit, y_fit, color='black', marker='x', s=40, label=f'{label} quad fit points')

                def quad_func(x, a, b, c):
                    return a * x**2 + b * x + c

                try:



                    popt, pcov = curve_fit(quad_func, x_fit, y_fit, absolute_sigma=True, p0 = [0.1,0,10])
                    x_plot = np.linspace(min(x_fit), max(x_fit), 100)
                    y_plot = quad_func(x_plot, *popt)
                    ax.plot(x_plot, y_plot, '--', color="k", alpha=0.7, label=f'{label} quad fit')
                    # Estimate minimum x value (vertex of parabola)
                    a, b, c = popt
                    if a != 0:
                        x_min = -b / (2 * a)
                        # Error estimate for x_min
                        # propagate error: dx = sqrt((db/(2a))^2 + (b*da/(2a^2))^2)
                        da = np.sqrt(pcov[0, 0])
                        db = np.sqrt(pcov[1, 1])
                        dx_min = np.sqrt((db / (2 * a))**2 + (b * da / (2 * a**2))**2)
                        ax.axvline(x_min, color="k", linestyle=':', alpha=0.5, label=f'{label} min x={x_min:.2f}±{dx_min:.2f}')
                    print("optimal effort for n_spins: ", n_spins, " is at n_hops: ", x_min, " +/- ", dx_min)
                    if x_min < min(x_fit):
                        x_min = min(x_fit)
                        dx_min = min(x_fit)
                        

                    y_at_xmin = quad_func(x_min, *popt)
                        
                except Exception as e:
                    print(f"Quadratic fit failed for {label}: {e}")
                    x_min = np.nan
                    dx_min = np.nan
                    y_at_xmin = np.nan
            else:
                x_min = np.nan
                dx_min = np.nan
                y_at_xmin = np.nan
        else:
            x_min = np.nan
            dx_min = np.nan
            y_at_xmin = np.nan
        return x_min, dx_min, y_at_xmin
    
    x_mins = np.zeros((len(spins_to_plot), len(m_replicas_list), len(m_quantum_replicas_list), 3))  # Store x_min and dx_min for each subplot and results_list
    fig, axs = plt.subplots(len(n_spins_list), len(m_replicas_list)*len(m_quantum_replicas_list), figsize=(6*len(m_replicas_list)*len(m_quantum_replicas_list), 4*len(n_spins_list)))
    #fig, axs = plt.subplots(len(n_spins_list), len(m_replicas_list), figsize=(6*len(m_replicas_list), 4*len(n_spins_list)))

    count = 0
    print("n_spins_to_plot:", spins_to_plot)
    for j, n_spins_to_plot in enumerate(spins_to_plot):
        for k, m_replica in enumerate(m_replicas_list):
            for l, m_quantum_replica in enumerate(m_quantum_replicas_list):


                if len(n_spins_list) >1 and  len(m_replicas_list)*len(m_quantum_replicas_list) > 1:
                    ax = axs[j, l]
                elif len(n_spins_list) >1 and len(m_replicas_list)*len(m_quantum_replicas_list) <1.1:
                    ax = axs[j]
                else:
                    ax = axs
                

                data = results['data_save'][count]
                x_min, dx_min, y_min = process_and_plot(ax, data, str(m_quantum_replica), colormap(l), n_spins_to_plot)
                x_mins[j,k,l,:] = [x_min, dx_min, y_min]
                count+=1
                
                ax.set_yscale( "log")
                ax.set_ylabel('Effort')
                ax.set_xlabel('Number of Hops')
                ax.set_title(f'Effort vs Hops for {n_spins_to_plot} Spins, m_replica={m_replica}')
                
                # Force the y-axis to be 10% larger than the data points only
                #padding = (np.max(data) - np.min(data)) * 0.1
                #plt.ylim(np.min(data) - padding, np.max(data) + padding)

        """row, col = divmod(j, n_cols)
        ax = axs[row][col]
        for i in range(len(results_list)):
            data = results_list[n_spins_list_index[i]]['data_save'][j]
            x_min, dx_min = process_and_plot(ax, data, results_labels[n_spins_list_index[i]], colors[n_spins_list_index[i]], n_spins_to_plot)
            x_mins[j,i,:] = [x_min, dx_min]
        ax.set_yscale("log")
        ax.set_ylabel('Effort')
        ax.set_xlabel('Number of Hops')
        ax.set_title(f'Effort vs Hops for {n_spins_to_plot} Spins')
        """





    fig.tight_layout()
    plt.savefig(output_dir / f'effort_vs_nhops_{n_spins_to_plot}.png')
    plt.close(fig)
    
    
    # Plot x_min vs n_spins for each results_list
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 5))
    x_mins = np.array(x_mins)

    from matplotlib.colors import LinearSegmentedColormap

    # Define the number of steps based on your replica lists
    num_colors =  len(m_quantum_replicas_list) 

    # Create a custom colormap from lightblue to lightgreen
    colors = ["lightgreen", "lightblue"]
    cmap = LinearSegmentedColormap.from_list("custom_gb", colors, N=num_colors)
    # Subplot 1: Optimal nhops (x_min) vs n_spins
    for i, m_replica in enumerate(m_replicas_list):
        for j, m_quantum_replica in enumerate(m_quantum_replicas_list):
            colour = cmap(i * len(m_quantum_replicas_list) + j)
            ax2.errorbar(spins_to_plot, x_mins[:, i, j, 0], yerr=x_mins[:, i, j, 1],
                         label=f'$M_q={m_quantum_replica}, M={m_replica}$', marker=markerslist[j], markersize=markersizelist[j], linewidth = 0.6, color = colour)
    
    ax2.set_xlabel('Number of Spins')
    ax2.set_ylabel('Optimal number of steps')
    ax2.set_title(f'Scaling of optimal number of steps for (Qe)PT | $m = {m}$')
    ax2.legend()
    # ax2.set_yscale('log')

    # Subplot 2: Optimal effort vs n_spins
    for i, m_replica in enumerate(m_replicas_list):
        for j, m_quantum_replica in enumerate(m_quantum_replicas_list):
            colour = cmap(i * len(m_quantum_replicas_list) + j)
            ax3.plot(spins_to_plot, x_mins[:, i, j, 2],
                         label=f'$M_q$={m_quantum_replica}', marker=markerslist[j],  linewidth = 0.6, color=colour, markersize=markersizelist[j])
    ax3.set_xlabel('Number of spins')
    ax3.set_ylabel('Optimal effort')
    ax3.set_title(f'Optimal effort vs number of spins | $m = {m}$')
    ax3.legend()
    ax3.set_yscale('log')

    fig2.tight_layout()
    fig2.savefig(output_dir / 'PT_nhops_vs_n.png')
    plt.close(fig2)
    fig3.tight_layout()
    fig3.savefig(output_dir / f'PT_optimal_effort_vs_n_spins_m_{m}.png')
    plt.close(fig3)

    # save x_mins to pickle
    pickle.dump({"x_mins": x_mins,
                "m_quantum_replicas_list": m_quantum_replicas_list,
                "m_replicas_list": m_replicas_list, 
                "spins_to_plot": spins_to_plot,
                 }, open(output_dir / 'nspins_m_mq__xmin_dxmin_ymin__.pkl', 'wb'))



def plot_hops_vs_probability(n_spins_list, m_replicas_list, m_quantum_replicas_list, results, output_dir, color):
    """
    Plots n_hops vs success probability for selected n_spins.
    """
    spins_to_plot = n_spins_list
    
    
    # Plot and save 1 - success probability for each n_spins
    fig_prob, ax_prob = plt.subplots(len(spins_to_plot), len(m_replicas_list) * len(m_quantum_replicas_list), figsize=(5*len(spins_to_plot), 5*len(m_replicas_list) * len(m_quantum_replicas_list)), sharey=True)
    count = 0
    for j, n_spins_to_plot in enumerate(spins_to_plot):
        print("n_spins_to_plot:", n_spins_to_plot)
        for k, m_replica in enumerate(m_replicas_list):
            print("m_replica:", m_replica)
            for l, m_quantum_replica in enumerate(m_quantum_replicas_list):
                print("m_quantum_replica:", m_quantum_replica)
                    
                if len(n_spins_list) >1:
                    ax = ax_prob[j, k*len(m_quantum_replicas_list)+l]
                else:
                    ax = ax_prob[k*len(m_quantum_replicas_list)+l]
                
                data = results['data_save'][count]
                if data is not None and isinstance(data, np.ndarray) and data.shape[1] >= 4:
                    nhops = data[:, 0]
                    probs = data[:, 3]

                    label = f'm_replica={m_replica}, m_q={m_quantum_replica}'
                    unique_nhops = np.unique(nhops)
                    mean_probs = []
                    sem_probs = []

                    for nhop in unique_nhops:
                        probs_at_hops = probs[nhops == nhop]
                        mean_probs.append(np.mean(probs_at_hops))
                        sem_probs.append(np.std(probs_at_hops, ddof=1) / np.sqrt(len(probs_at_hops)) if len(probs_at_hops) > 1 else 0)
                        #ax.scatter(np.ones(len(hops_at_effort))*effort, hops_at_effort, marker='x', color=color, s=10, alpha = 0.1)
                    ax.errorbar(unique_nhops, mean_probs, yerr=sem_probs, label=label, marker="o", linestyle="", markersize=4, color="k")
                


                    ax.scatter(nhops, probs, label="scatter", marker="o", s=10, alpha = 0.5, color=color)

                ax.set_yscale("log")
                ax.set_ylabel('probs')
                ax.set_xlabel('Number of Hops')
                ax.set_title(f'{n_spins_to_plot} Spins, m_replica={m_replica}, m_q={m_quantum_replica}')
                

                count+=1
    #fig.legend()
    fig_prob.supylabel('Success Probability')
    fig_prob.supxlabel('Number of Hops')
    fig_prob.legend()
    fig_prob.tight_layout()
    fig_prob.savefig(output_dir / 'hops_vs_probability.png')
    plt.close(fig_prob)
    
    
    
    

if __name__ == "__main__":
    
    
    dir_ = Path(__file__).resolve().parent
    print("dir_", dir_)
    results_dir = dir_ / "results_PT"
    print(results_dir)
    results_path = dir_ / "plots_PT"
    results_path.mkdir(parents=True, exist_ok=True)
    n_spins_list = [5,6,7,8,9,10,12,15,20]
    m_replicas_list = [4,]#np.arange(2,17)
    m_quantum_replicas_list = [0,1]#,1,2]
    m = 2
    proposals_list = [["local",]*(m_replicas_list[i]-m_quantum_replicas_list[j]) + ["qemcmc"]*m_quantum_replicas_list[j] for i in range(len(m_replicas_list)) for j in range(len(m_quantum_replicas_list))]

    print("n_spins_list:", n_spins_list)
    print("m_replicas_list:", m_replicas_list)
    print("m_quantum_replicas_list:", m_quantum_replicas_list)
    print("proposals_list:", proposals_list)
    #main(results_dir, results_path,n_spins_list,m_replicas_list)

    
    #pt_nhops, pt_sem_nhops, pt_efforts, pt_sem_efforts = load_pt_results(n_spins_list, m_replicas_list, results_dir, m_quantum_replicas_list)
    #plot_optimal_m_replicas(pt_efforts, pt_sem_efforts)
    #plot_optimal_effort_n_spins_m_const(pt_efforts, pt_sem_efforts,4,n_spins_list, m_replicas_list, m_quantum_replicas_list )
    
    cmap = plt.get_cmap('inferno', len(m_replicas_list) * len(m_quantum_replicas_list)+2)

    results = load_pt_results(n_spins_list, m_replicas_list, results_dir, proposals_list=proposals_list, m = m)
    plot_effort_vs_nhops(n_spins_list, m_replicas_list, m_quantum_replicas_list , results, results_path, "", cmap)
    plot_hops_vs_probability(n_spins_list, m_replicas_list, m_quantum_replicas_list, results, results_path, "lightgreen")
