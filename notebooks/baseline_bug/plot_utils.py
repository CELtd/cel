import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mechafil_jax.date_utils as du
from matplotlib.lines import Line2D

import os
colors = [mpl.colormaps['Blues'], mpl.colormaps['Oranges']]

def get_quantiles(jax_arr, qvec=[0.05, 0.25, 0.5, 0.75, 0.95]):
    return np.quantile(jax_arr, qvec, axis=0)

def plot_inputs(mcmc_trajectories, hist_inputs, current_date, save_fp):
    # plot historical trends
    forecast_length = mcmc_trajectories.rb_onboard_power_pred.shape[1]
    t_pred = pd.date_range(current_date, periods=forecast_length, freq='D')

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3), sharex=True)

    blues = mpl.colormaps['Blues']
    axx = ax[0]
    axx.plot(hist_inputs.hist_plot_tvec_rbp, hist_inputs.hist_rbp, color='k')
    rbp_quantiles = get_quantiles(mcmc_trajectories.rb_onboard_power_pred)
    axx.fill_between(t_pred, rbp_quantiles[0], rbp_quantiles[4], color=blues(0.2))
    axx.fill_between(t_pred, rbp_quantiles[1], rbp_quantiles[3], color=blues(0.5))
    axx.plot(t_pred, rbp_quantiles[2], color=blues(0.9))
    axx.set_title('Onboarding')
    axx.set_ylabel('RBP/day')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
        
    axx = ax[1]
    axx.plot(hist_inputs.hist_plot_tvec_rr, hist_inputs.hist_rr*100, color='k')
    rr_quantiles = get_quantiles(mcmc_trajectories.renewal_rate_pred*100)
    axx.fill_between(t_pred, rr_quantiles[0], rr_quantiles[4], color=blues(0.2))
    axx.fill_between(t_pred, rr_quantiles[1], rr_quantiles[3], color=blues(0.5))
    axx.plot(t_pred, rr_quantiles[2], color=blues(0.9))
    axx.set_title('Renewals')
    axx.set_ylabel('%')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)

    axx = ax[2]
    axx.plot(hist_inputs.hist_plot_tvec_fpr, hist_inputs.hist_fpr*100, color='k')
    fpr_quantiles = get_quantiles(mcmc_trajectories.filplus_rate_pred*100)
    axx.fill_between(t_pred, fpr_quantiles[0], fpr_quantiles[4], color=blues(0.2))
    axx.fill_between(t_pred, fpr_quantiles[1], fpr_quantiles[3], color=blues(0.5))
    axx.plot(t_pred, fpr_quantiles[2], color=blues(0.9))
    axx.set_title('FIL+')
    axx.set_ylabel('%')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)

    plt.tight_layout()
    plt.savefig(save_fp)

def get_simresults_quantiles(sim_results_vec, key, qvec=[0.05, 0.25, 0.5, 0.75, 0.95]):
    key_results = []
    for sr in sim_results_vec:
        key_results.append(np.asarray(sr[key]))  # convert jax to np
    
    return np.nanquantile(np.asarray(key_results), qvec, axis=0)

def plot_mcmc_power_panel(hist_kpi_df, simulation_results_vec, start_date, current_date, end_date, vlines, vline_labels, save_fp):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))

    macro_t = du.get_t(start_date, end_date=end_date)

    axx = ax[0]
    alpha_vec = np.ones(len(simulation_results_vec)) / len(simulation_results_vec)
    for simulation_results, color, alpha in zip(simulation_results_vec, colors, alpha_vec):
        yy = get_simresults_quantiles(simulation_results, key='rb_total_power_eib')
        axx.plot(hist_kpi_df['date'], hist_kpi_df['total_raw_power_eib'], color='k')
        axx.fill_between(macro_t, yy[0], yy[4], color=color(0.2), alpha=alpha)
        axx.fill_between(macro_t, yy[1], yy[3], color=color(0.5), alpha=alpha)
        axx.plot(macro_t, yy[2], color=color(0.9))
    axx.set_ylabel('EiB')
    axx.set_title('RBP')
    # axx.legend(fontsize=8)
    axx.set_ylim(bottom=0)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    greys = mpl.colormaps['Greys']
    cvec_idx = np.linspace(0.9, 0.2, len(vlines))
    for vline, vline_label, cidx in zip(vlines, vline_labels, cvec_idx):
        axx.axvline(vline, color=greys(cidx), linestyle='--', label=vline_label)
    axx.axvline(current_date, color='k', linestyle=':', linewidth=0.5, label='Forecast Start')
    axx.legend()
    
    axx = ax[1]
    for simulation_results, color, alpha in zip(simulation_results_vec, colors, alpha_vec):
        yy = get_simresults_quantiles(simulation_results, key='qa_total_power_eib')
        axx.plot(hist_kpi_df['date'], hist_kpi_df['qa_total_power_eib'], color='k')
        axx.fill_between(macro_t, yy[0], yy[4], color=color(0.2), alpha=alpha)
        axx.fill_between(macro_t, yy[1], yy[3], color=color(0.5), alpha=alpha)
        axx.plot(macro_t, yy[2], color=color(0.9))
    baseline = get_simresults_quantiles(simulation_results, key='network_baseline_EIB', qvec=[0.5])
    axx.plot(macro_t, baseline[0], color='k', linestyle='--', label='Baseline')
    axx.set_ylabel('EiB')
    axx.set_title('QAP')
    axx.legend(fontsize=8)
    axx.set_ylim(bottom=0)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    greys = mpl.colormaps['Greys']
    cvec_idx = np.linspace(0.9, 0.2, len(vlines))
    for vline, vline_label, cidx in zip(vlines, vline_labels, cvec_idx):
        axx.axvline(vline, color=greys(cidx), linestyle='--', label=vline_label)
    axx.axvline(current_date, color='k', linestyle=':', linewidth=0.5, label='Forecast Start')
    
    axx = ax[2]
    for simulation_results, color, alpha in zip(simulation_results_vec, colors, alpha_vec):
        yy = get_simresults_quantiles(simulation_results, key='day_network_reward')
        axx.plot(hist_kpi_df['date'], hist_kpi_df['day_network_reward'], color='k')
        axx.fill_between(macro_t, yy[0], yy[4], color=color(0.2), alpha=alpha)
        axx.fill_between(macro_t, yy[1], yy[3], color=color(0.5), alpha=alpha)
        axx.plot(macro_t, yy[2], color=color(0.9))
    axx.set_ylabel('FIL/day')
    axx.set_title('Minting Rate')
    # axx.legend(fontsize=8)
    axx.set_ylim(bottom=0)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    greys = mpl.colormaps['Greys']
    cvec_idx = np.linspace(0.9, 0.2, len(vlines))
    for vline, vline_label, cidx in zip(vlines, vline_labels, cvec_idx):
        axx.axvline(vline, color=greys(cidx), linestyle='--', label=vline_label)
    axx.axvline(current_date, color='k', linestyle=':', linewidth=0.5, label='Forecast Start')
    
    plt.tight_layout()
    plt.savefig(save_fp)

def plot_mcmc_supply_panel(hist_kpi_df, simulation_results_vec, start_date, current_date, end_date, vlines, vline_labels, save_fp):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))

    macro_t = du.get_t(start_date, end_date=end_date)
    axx = ax[0]
    alpha_vec = np.ones(len(simulation_results_vec)) / len(simulation_results_vec)
    for simulation_results, color, alpha in zip(simulation_results_vec, colors, alpha_vec):
        yy = get_simresults_quantiles(simulation_results, key='network_locked')/1e6
        axx.plot(hist_kpi_df['date'], hist_kpi_df['network_locked']/1e6, color='k')
        axx.fill_between(macro_t, yy[0], yy[4], color=color(0.2), alpha=alpha)
        axx.fill_between(macro_t, yy[1], yy[3], color=color(0.5), alpha=alpha)
        axx.plot(macro_t, yy[2], color=color(0.9))
    axx.set_ylabel('M-FIL')
    axx.set_title('Network Locked')
    axx.set_ylim(bottom=0)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    greys = mpl.colormaps['Greys']
    cvec_idx = np.linspace(0.9, 0.2, len(vlines))
    for vline, vline_label, cidx in zip(vlines, vline_labels, cvec_idx):
        axx.axvline(vline, color=greys(cidx), linestyle='--', label=vline_label)
    axx.axvline(current_date, color='k', linestyle=':', linewidth=0.5, label='Forecast Start')
    axx.axhline(100/3, color='red', linestyle='--', alpha=0.5, label='100M-USD TVL @$3/FIL')
    axx.legend(fontsize=8, loc='upper right')
    
    axx = ax[1]
    alpha_vec = np.ones(len(simulation_results_vec)) / len(simulation_results_vec)
    for simulation_results, color, alpha in zip(simulation_results_vec, colors, alpha_vec):
        yy = get_simresults_quantiles(simulation_results, key='circ_supply')/1e6
        axx.plot(hist_kpi_df['date'], hist_kpi_df['circ_supply']/1e6, color='k')
        axx.fill_between(macro_t, yy[0], yy[4], color=color(0.2), alpha=alpha)
        axx.fill_between(macro_t, yy[1], yy[3], color=color(0.5), alpha=alpha)
        axx.plot(macro_t, yy[2], color=color(0.9))
    axx.set_ylabel('M-FIL')
    axx.set_title('Circulating Supply')
    # axx.legend(fontsize=8)
    axx.set_ylim(bottom=0)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    axx.axvline(current_date, color='grey', linestyle='--')
    greys = mpl.colormaps['Greys']
    cvec_idx = np.linspace(0.9, 0.2, len(vlines))
    for vline, vline_label, cidx in zip(vlines, vline_labels, cvec_idx):
        axx.axvline(vline, color=greys(cidx), linestyle='--', label=vline_label)
    axx.axvline(current_date, color='k', linestyle=':', linewidth=0.5, label='Forecast Start')
    
    axx = ax[2]
    alpha_vec = np.ones(len(simulation_results_vec)) / len(simulation_results_vec)
    for simulation_results, color, alpha in zip(simulation_results_vec, colors, alpha_vec):
        key_results = []
        for sr in simulation_results:
            key_results.append(np.asarray(sr['network_locked']/sr['circ_supply']))  # convert jax to np
        lcs_q = np.nanquantile(np.asarray(key_results), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)*100
        
        axx.plot(hist_kpi_df['date'], hist_kpi_df['network_locked']/hist_kpi_df['circ_supply']*100, color='k')
        axx.fill_between(macro_t, lcs_q[0], lcs_q[4], color=color(0.2), alpha=alpha)
        axx.fill_between(macro_t, lcs_q[1], lcs_q[3], color=color(0.5), alpha=alpha)
        axx.plot(macro_t, lcs_q[2], color=color(0.9))
    axx.set_ylabel('%')
    axx.set_title('L/CS')
    axx.set_ylim(bottom=0)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    axx.axvline(current_date, color='grey', linestyle='--')
    axx.axhline(30, color='red', linestyle='--', alpha=0.5, label='Lock Target')
    greys = mpl.colormaps['Greys']
    cvec_idx = np.linspace(0.9, 0.2, len(vlines))
    for vline, vline_label, cidx in zip(vlines, vline_labels, cvec_idx):
        axx.axvline(vline, color=greys(cidx), linestyle='--', label=vline_label)
    axx.axvline(current_date, color='k', linestyle=':', linewidth=0.5, label='Forecast Start')
    axx.legend(fontsize=8)
    
    plt.suptitle('Supply Metrics')
    plt.tight_layout()
    plt.savefig(save_fp)

def plot_mcmc_supply_panel_delta(simulation_results_vec, start_date, current_date, end_date, save_fp):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
    blues = mpl.colormaps['Blues']

    macro_t = du.get_t(start_date, end_date=end_date)
    axx = ax[0]
    yy = get_simresults_quantiles(simulation_results_vec, key='network_locked')
    axx.fill_between(macro_t, yy[0], yy[4], color=blues(0.2))
    axx.fill_between(macro_t, yy[1], yy[3], color=blues(0.5))
    axx.plot(macro_t, yy[2], color=blues(0.9))
    axx.set_ylabel('%')
    axx.set_title(r'$\Delta Network Locked$')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    axx.axvline(current_date, color='grey', linestyle='--')
    axx.legend(fontsize=8, loc='upper right')
    
    axx = ax[1]
    yy = get_simresults_quantiles(simulation_results_vec, key='circ_supply')
    axx.fill_between(macro_t, yy[0], yy[4], color=blues(0.2))
    axx.fill_between(macro_t, yy[1], yy[3], color=blues(0.5))
    axx.plot(macro_t, yy[2], color=blues(0.9))
    axx.set_ylabel('%')
    axx.set_title(r'$\Delta Circulating Supply$')
    # axx.legend(fontsize=8)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    key_results = []
    for sr in simulation_results_vec:
        # div by zero when computing delta
        lcs = np.nan_to_num(np.asarray(sr['network_locked']/sr['circ_supply']), nan=0)
        key_results.append(lcs)  # convert jax to np
    lcs_q = np.nanquantile(np.asarray(key_results), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)*100
    axx = ax[2]
    axx.fill_between(macro_t, lcs_q[0], lcs_q[4], color=blues(0.2))
    axx.fill_between(macro_t, lcs_q[1], lcs_q[3], color=blues(0.5))
    axx.plot(macro_t, lcs_q[2], color=blues(0.9))
    axx.set_ylabel('%')
    axx.set_title('L/CS')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    axx.axvline(current_date, color='grey', linestyle='--')
    axx.legend(fontsize=8)
    
    plt.suptitle('Supply Metrics')
    plt.tight_layout()
    plt.savefig(save_fp)

def plot_mcmc_onboarding_panel(hist_kpi_df, simulation_results_vec, start_date, current_date, end_date, vlines, vline_labels, save_fp):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    macro_t = du.get_t(start_date, end_date=end_date)
    axx = ax[0]
    alpha_vec = np.ones(len(simulation_results_vec)) / len(simulation_results_vec)
    for simulation_results, color, alpha in zip(simulation_results_vec, colors, alpha_vec):
        yy = get_simresults_quantiles(simulation_results, key='day_pledge_per_QAP')
        axx.plot(hist_kpi_df['date'], hist_kpi_df['day_pledge_per_QAP'], color='k')
        axx.fill_between(macro_t, yy[0], yy[4], color=color(0.2), alpha=alpha)
        axx.fill_between(macro_t, yy[1], yy[3], color=color(0.5), alpha=alpha)
        axx.plot(macro_t, yy[2], color=color(0.9))
    axx.set_ylabel('FIL')
    axx.set_title('Pledge/32GiB QA Sector')
    # axx.legend(fontsize=8)
    axx.set_ylim(bottom=0)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    axx = ax[1]
    alpha_vec = np.ones(len(simulation_results_vec)) / len(simulation_results_vec)
    for simulation_results, color, alpha in zip(simulation_results_vec, colors, alpha_vec):
        key_results = []
        for sr in simulation_results:
            # we need to create new time-series and append historical to forecast to compute a smooth FoFR plot
            dpqq_full = np.concatenate([hist_kpi_df['day_pledge_per_QAP'].values, sr['day_pledge_per_QAP']])
            drps_full = np.concatenate([hist_kpi_df['day_rewards_per_sector'].values, sr['day_rewards_per_sector']])
            days_1y = 365
            rps_1y = np.convolve(drps_full, np.ones(days_1y), mode='full')[days_1y-1:1-days_1y]
            roi_1y = rps_1y / dpqq_full[:1-days_1y]
            key_results.append(np.asarray(roi_1y*100))

        fofr = np.nanquantile(np.asarray(key_results), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        fofr_tvec = pd.date_range(hist_kpi_df['date'].iloc[0], periods=len(fofr[0]), freq='D')
        axx.fill_between(fofr_tvec, fofr[0], fofr[4], color=color(0.2), alpha=alpha)
        axx.fill_between(fofr_tvec, fofr[1], fofr[3], color=color(0.5), alpha=alpha)
        axx.plot(fofr_tvec, fofr[2], color=color(0.9))

    axx.set_ylabel('%')
    axx.set_title('1Y Realized FoFR')
    axx.set_ylim(bottom=0)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    plt.suptitle('Onboarding Metrics')
    plt.tight_layout()

    plt.savefig(save_fp)

def plot_mcmc_onboarding_panel_delta(simulation_results_vec, start_date, current_date, end_date, save_fp):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    blues = mpl.colormaps['Blues']

    macro_t = du.get_t(start_date, end_date=end_date)
    axx = ax[0]
    yy = get_simresults_quantiles(simulation_results_vec, key='day_pledge_per_QAP')
    axx.fill_between(macro_t, yy[0], yy[4], color=blues(0.2))
    axx.fill_between(macro_t, yy[1], yy[3], color=blues(0.5))
    axx.plot(macro_t, yy[2], color=blues(0.9))
    axx.set_ylabel('%')
    axx.set_title(r'$\Delta Pledge/32GiB QA Sector$')
    # axx.legend(fontsize=8)
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    # axx = ax[1]
    # key_results = []
    # for sr in simulation_results_vec:
    #     # we need to create new time-series and append historical to forecast to compute a smooth FoFR plot
    #     dpqq_full = np.concatenate([hist_kpi_df['day_pledge_per_QAP'].values, sr['day_pledge_per_QAP']])
    #     drps_full = np.concatenate([hist_kpi_df['day_rewards_per_sector'].values, sr['day_rewards_per_sector']])
    #     days_1y = 365
    #     rps_1y = np.convolve(drps_full, np.ones(days_1y), mode='full')[days_1y-1:1-days_1y]
    #     roi_1y = rps_1y / dpqq_full[:1-days_1y]
    #     key_results.append(np.asarray(roi_1y*100))

    # fofr = np.nanquantile(np.asarray(key_results), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    # fofr_tvec = pd.date_range(hist_kpi_df['date'].iloc[0], periods=len(fofr[0]), freq='D')
    # axx.fill_between(fofr_tvec, fofr[0], fofr[4], color=blues(0.2))
    # axx.fill_between(fofr_tvec, fofr[1], fofr[3], color=blues(0.5))
    # axx.plot(fofr_tvec, fofr[2], color=blues(0.9))

    # axx.set_ylabel('%')
    # axx.set_title('1Y Realized FoFR')
    # axx.set_ylim(bottom=0)
    # for tick in axx.get_xticklabels(): tick.set_rotation(60)
    # axx.axvline(current_date, color='grey', linestyle='--')
    
    plt.suptitle(r'\Delta Onboarding Metrics')
    plt.tight_layout()

    plt.savefig(save_fp)

def plot_locked_value_distribution(filprice2lvd, tvl_target, save_fp):
    qvec = [0.05, 0.25, 0.5, 0.75, 0.95]
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
    oranges = mpl.colormaps['Oranges']
    cvec_idx = np.linspace(0.9, 0.2, len(qvec))
    ii = 0
    for fil_price, lvd in filprice2lvd.items():
        axx = ax[ii]

        date_quantiles = np.quantile(lvd, qvec, axis=0)
        axx.hist(lvd, bins=50)
        # plot the quantiles as vertical lines
        jj = 0
        for q,dq in zip(qvec, date_quantiles):
            axx.axvline(dq, color=oranges(cvec_idx[jj]), linestyle='--', label='Q-%d: %s' % (q*100,dq.strftime('%Y-%m-%d')))
            jj += 1

        axx.set_xlabel('Date')
        axx.set_title('%d USD/FIL' % (fil_price,))
        for tick in axx.get_xticklabels(): tick.set_rotation(60)
        axx.legend(fontsize=8)
        ii += 1
    
    plt.suptitle('Crossing Date for TVL Target=%dM-USD' % (tvl_target/1e6,))
    plt.tight_layout()
    plt.savefig(save_fp)

def plot_power_scenarios(hist_kpi_df, results_dict, start_date, current_date, end_date, rbp_factors, rr_factors, fpr_factors, save_fp):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))

    macro_t = du.get_t(start_date, end_date=end_date)
    colors = [mpl.colormaps['Blues'], mpl.colormaps['Oranges']]
    intensities = np.linspace(0.3, 0.7, len(rr_factors))
    linestyles = ['-.', ':']
    
    axx = ax[0]
    for sim_config, sim_results in results_dict.items():
        rbp_factor, rr_factor, fpr_factor = sim_config
        if rbp_factor == 1 and rr_factor == 1 and fpr_factor == 1:
            c = 'k'
            l='-'
        else:
            c = colors[rbp_factors.index(rbp_factor)](intensities[rr_factors.index(rr_factor)])
            l = linestyles[fpr_factors.index(fpr_factor)]
            
        axx.plot(macro_t, sim_results['rb_total_power_eib'], color=c, linestyle=l)
    axx.plot(hist_kpi_df['date'], hist_kpi_df['total_raw_power_eib'], color='k')
    axx.set_ylabel('EiB')
    axx.set_title('RBP')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    # setup the legend
    custom_lines = [
        Line2D([0], [0], color=colors[0](0.3), lw=2),
        Line2D([0], [0], color='k', lw=2),
        Line2D([0], [0], color=colors[1](0.7), lw=2)
    ]
    axx.legend(custom_lines, ['0.8x', '1x', '1.2x'], title='RBP Factor')
    axx.set_ylim(bottom=0)
    axx.axvline(current_date, color='grey', linestyle='--')

    axx = ax[1]
    for sim_config, sim_results in results_dict.items():
        rbp_factor, rr_factor, fpr_factor = sim_config
        if rbp_factor == 1 and rr_factor == 1 and fpr_factor == 1:
            c = 'k'
            l='-'
        else:
            c = colors[rbp_factors.index(rbp_factor)](intensities[rr_factors.index(rr_factor)])
            l = linestyles[fpr_factors.index(fpr_factor)]
            
        axx.plot(macro_t, sim_results['qa_total_power_eib'], color=c, linestyle=l)
    axx.plot(hist_kpi_df['date'], hist_kpi_df['qa_total_power_eib'], color='k')
    axx.plot(macro_t, sim_results['network_baseline_EIB'], color='k', linestyle='--')
    axx.set_ylabel('EiB')
    axx.set_title('QAP')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    # setup the legend
    # setup the legend
    custom_lines = [
        Line2D([0], [0], color='k', lw=2, linestyle='-.'),
        Line2D([0], [0], color='k', lw=2),
        Line2D([0], [0], color='k', lw=2, linestyle=':'),
    ]
    axx.legend(custom_lines, ['0.8x', '1x', '1.2x'], title='FPR Factor')
    axx.set_ylim(bottom=0)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    axx = ax[2]
    for sim_config, sim_results in results_dict.items():
        rbp_factor, rr_factor, fpr_factor = sim_config
        if rbp_factor == 1 and rr_factor == 1 and fpr_factor == 1:
            c = 'k'
            l='-'
        else:
            c = colors[rbp_factors.index(rbp_factor)](intensities[rr_factors.index(rr_factor)])
            l = linestyles[fpr_factors.index(fpr_factor)]
            
        axx.plot(macro_t, sim_results['day_network_reward'], color=c, linestyle=l)
    axx.plot(hist_kpi_df['date'], hist_kpi_df['day_network_reward'], color='k')
    axx.set_ylabel('FIL/day')
    axx.set_title('Minting Rate')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    custom_lines = [
        Line2D([0], [0], color=mpl.colormaps['Greys'](intensities[0]), lw=2),
        Line2D([0], [0], color='k', lw=2),
        Line2D([0], [0], color=mpl.colormaps['Greys'](intensities[1]), lw=2),
    ]
    axx.legend(custom_lines, ['0.8x', '1x', '1.2x'], title='RR Factor')
    axx.set_ylim(bottom=0)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    # plt.suptitle('Power Metrics')
    plt.tight_layout()

    plt.savefig(save_fp)

def plot_supply_scenarios(hist_kpi_df, results_dict, start_date, current_date, end_date, rbp_factors, rr_factors, fpr_factors, save_fp):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))

    macro_t = du.get_t(start_date, end_date=end_date)
    colors = [mpl.colormaps['Blues'], mpl.colormaps['Oranges']]
    intensities = np.linspace(0.3, 0.7, len(rr_factors))
    linestyles = ['-.', ':']
    
    axx = ax[0]
    for sim_config, sim_results in results_dict.items():
        rbp_factor, rr_factor, fpr_factor = sim_config
        if rbp_factor == 1 and rr_factor == 1 and fpr_factor == 1:
            c = 'k'
            l='-'
        else:
            c = colors[rbp_factors.index(rbp_factor)](intensities[rr_factors.index(rr_factor)])
            l = linestyles[fpr_factors.index(fpr_factor)]
            
        axx.plot(macro_t, sim_results['network_locked']/1e6, color=c, linestyle=l)
    axx.plot(hist_kpi_df['date'], hist_kpi_df['network_locked']/1e6, color='k')
    axx.set_ylabel('M-FIL')
    axx.set_title('Network Locked')
    axx.axhline(100/3, color='red', linestyle='--', alpha=0.5, label='100M-USD TVL @$3/FIL')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    # setup the legend
    custom_lines = [
        Line2D([0], [0], color=colors[0](0.3), lw=2),
        Line2D([0], [0], color='k', lw=2),
        Line2D([0], [0], color=colors[1](0.7), lw=2)
    ]
    axx.legend(custom_lines, ['0.8x', '1x', '1.2x'], title='RBP Factor')
    axx.set_ylim(bottom=0)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    axx = ax[1]
    for sim_config, sim_results in results_dict.items():
        rbp_factor, rr_factor, fpr_factor = sim_config
        if rbp_factor == 1 and rr_factor == 1 and fpr_factor == 1:
            c = 'k'
            l='-'
        else:
            c = colors[rbp_factors.index(rbp_factor)](intensities[rr_factors.index(rr_factor)])
            l = linestyles[fpr_factors.index(fpr_factor)]
            
        axx.plot(macro_t, sim_results['circ_supply']/1e6, color=c, linestyle=l)
    axx.plot(hist_kpi_df['date'], hist_kpi_df['circ_supply']/1e6, color='k')
    axx.set_ylabel('M-FIL')
    axx.set_title('Circulating Supply')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    # setup the legend
    custom_lines = [
        Line2D([0], [0], color='k', lw=2, linestyle='-.'),
        Line2D([0], [0], color='k', lw=2),
        Line2D([0], [0], color='k', lw=2, linestyle=':'),
    ]
    axx.legend(custom_lines, ['0.8x', '1x', '1.2x'], title='FPR Factor')
    axx.set_ylim(bottom=0)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    axx = ax[2]
    for sim_config, sim_results in results_dict.items():
        rbp_factor, rr_factor, fpr_factor = sim_config
        if rbp_factor == 1 and rr_factor == 1 and fpr_factor == 1:
            c = 'k'
            l='-'
        else:
            c = colors[rbp_factors.index(rbp_factor)](intensities[rr_factors.index(rr_factor)])
            l = linestyles[fpr_factors.index(fpr_factor)]
            
        axx.plot(macro_t, sim_results['network_locked']/sim_results['circ_supply']*100, color=c, linestyle=l)
    axx.plot(hist_kpi_df['date'], hist_kpi_df['network_locked']/hist_kpi_df['circ_supply']*100, color='k')
    axx.set_ylabel('%%')
    axx.set_title('L/CS')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    custom_lines = [
        Line2D([0], [0], color=mpl.colormaps['Greys'](intensities[0]), lw=2),
        Line2D([0], [0], color='k', lw=2),
        Line2D([0], [0], color=mpl.colormaps['Greys'](intensities[1]), lw=2),
    ]
    axx.legend(custom_lines, ['0.8x', '1x', '1.2x'], title='RR Factor')
    axx.set_ylim(bottom=0)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    plt.suptitle('Supply Metrics')
    plt.tight_layout()

    plt.savefig(save_fp)

def plot_onboarding_scenarios(hist_kpi_df, results_dict, start_date, current_date, end_date, rbp_factors, rr_factors, fpr_factors, save_fp):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    macro_t = du.get_t(start_date, end_date=end_date)
    colors = [mpl.colormaps['Blues'], mpl.colormaps['Oranges']]
    intensities = np.linspace(0.3, 0.7, len(rr_factors))
    linestyles = ['-.', ':']
    
    axx = ax[0]
    for sim_config, sim_results in results_dict.items():
        rbp_factor, rr_factor, fpr_factor = sim_config
        if rbp_factor == 1 and rr_factor == 1 and fpr_factor == 1:
            c = 'k'
            l='-'
        else:
            c = colors[rbp_factors.index(rbp_factor)](intensities[rr_factors.index(rr_factor)])
            l = linestyles[fpr_factors.index(fpr_factor)]
            
        axx.plot(macro_t, sim_results['day_pledge_per_QAP'], color=c, linestyle=l)
    axx.plot(hist_kpi_df['date'], hist_kpi_df['day_pledge_per_QAP'], color='k')
    axx.set_ylabel('FIL')
    axx.set_title('Pledge/32GiB QAP Sector')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    # setup the legend
    custom_lines = [
        Line2D([0], [0], color=colors[0](0.3), lw=2),
        Line2D([0], [0], color='k', lw=2),
        Line2D([0], [0], color=colors[1](0.7), lw=2)
    ]
    axx.legend(custom_lines, ['0.8x', '1x', '1.2x'], title='RBP Factor')
    axx.set_ylim(bottom=0)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    axx = ax[1]
    for sim_config, sim_results in results_dict.items():
        rbp_factor, rr_factor, fpr_factor = sim_config
        if rbp_factor == 1 and rr_factor == 1 and fpr_factor == 1:
            c = 'k'
            l='-'
        else:
            c = colors[rbp_factors.index(rbp_factor)](intensities[rr_factors.index(rr_factor)])
            l = linestyles[fpr_factors.index(fpr_factor)]

        # fofr = sim_results['1y_sector_roi']*100
        # axx.plot(macro_t[0:len(fofr)], fofr, color=c, linestyle=l)
        dpqq_full = np.concatenate([hist_kpi_df['day_pledge_per_QAP'].values, sim_results['day_pledge_per_QAP']])
        drps_full = np.concatenate([hist_kpi_df['day_rewards_per_sector'].values, sim_results['day_rewards_per_sector']])
        days_1y = 365
        rps_1y = np.convolve(drps_full, np.ones(days_1y), mode='full')[days_1y-1:1-days_1y]
        roi_1y = rps_1y / dpqq_full[:1-days_1y]
        fofr_tvec = pd.date_range(hist_kpi_df['date'].iloc[0], periods=len(roi_1y), freq='D')
        axx.plot(fofr_tvec, roi_1y*100, color=c, linestyle=l)
        
    axx.set_ylabel('%')
    axx.set_title('1Y FoFR')
    for tick in axx.get_xticklabels(): tick.set_rotation(60)
    custom_lines = [
        Line2D([0], [0], color='k', lw=2, linestyle='-.'),
        Line2D([0], [0], color='k', lw=2),
        Line2D([0], [0], color='k', lw=2, linestyle=':'),
    ]
    axx.legend(custom_lines, ['0.8x', '1x', '1.2x'], title='FPR Factor')
    axx.set_ylim(bottom=0)
    axx.axvline(current_date, color='grey', linestyle='--')
    
    plt.suptitle('Onboarding Metrics')
    plt.tight_layout()

    plt.savefig(save_fp)