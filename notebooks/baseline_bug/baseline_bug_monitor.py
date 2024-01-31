#!/usr/bin/env python
import numpyro
import argparse
import multiprocessing

import os
from datetime import date, timedelta
from tqdm.auto import tqdm
from collections import defaultdict
import itertools
from joblib import Memory
from types import SimpleNamespace 

import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax.random import PRNGKey

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import mechafil_jax.constants as C
import mechafil_jax.minting as minting
import mechafil_jax.plot_utils as pu
import mechafil_jax.date_utils as du

import scenario_generator.utils as u
import scenario_generator.curated as curated
import scenario_generator.mcmc_forecast as mcmc

import pystarboard.data
import yfinance as yf

import plot_utils as pu

cachedir = os.path.join(os.getcwd(), 'baseline_bug_monitor_cache')
memory = Memory(cachedir, verbose=0)
numpyro.set_platform('cpu')
numpyro.set_host_device_count(multiprocessing.cpu_count())

@memory.cache
def download_simulation_data(token, start_date, current_date, end_date):
    offline_data = data.get_simulation_data(token, start_date, current_date, end_date)
    return offline_data

@memory.cache
def generate_mcmc_forecast_samples(train_start_date: date,
                                   train_end_date: date,
                                   forecast_length: int,
                                   num_warmup_mcmc: int = 500,
                                   num_samples_mcmc: int = 100,
                                   seasonality_mcmc: int = 1000,
                                   num_chains_mcmc: int = 2,
                                   verbose: bool = False):
    if verbose: print("Forecasting Onboarding Power")
    _, rb_onboard_power_pred, historical_rb_date, historical_rb, rb_rhats = \
        mcmc.forecast_rb_onboard_power(train_start_date, 
                                       train_end_date,
                                       forecast_length,
                                       num_warmup_mcmc = num_warmup_mcmc,
                                       num_samples_mcmc = num_samples_mcmc,
                                       seasonality_mcmc = seasonality_mcmc,
                                       num_chains_mcmc = num_chains_mcmc,
                                       verbose = verbose)
    
    if verbose: print("Forecasting Renewal Rate")
    _, renewal_rate_pred, historical_rr_date , historical_rr, ext_rhats, expire_rhats = \
        mcmc.forecast_renewal_rate(train_start_date, 
                                   train_end_date,
                                   forecast_length,
                                   num_warmup_mcmc = num_warmup_mcmc,
                                   num_samples_mcmc = num_samples_mcmc,
                                   seasonality_mcmc = seasonality_mcmc,
                                   num_chains_mcmc = num_chains_mcmc,
                                   verbose = verbose)
    
    if verbose: print("Forecasting FIL+ Rate via SGT")
    _, filplus_rate_pred, historical_fpr_date, historical_fpr, deal_onboard_pred_rhats, cc_onboard_pred_rhats = \
        mcmc.forecast_filplus_rate(train_start_date, 
                                   train_end_date,
                                   forecast_length,
                                   num_warmup_mcmc = num_warmup_mcmc,
                                   num_samples_mcmc = num_samples_mcmc,
                                   seasonality_mcmc = seasonality_mcmc,
                                   num_chains_mcmc = num_chains_mcmc,
                                   verbose = verbose)

    ########################## TODO: decide if we want to do this ########################
    if verbose: print("Forecasting FIL+ Rate via Logistic Method")
    forecast_fpr_logistic_date_vec, filplus_rate_logistic_pred, _, _, fpr_rhat = \
        mcmc.forecast_filplus_rate_logistic(
            train_end_date,
            forecast_length,
            num_warmup_mcmc = num_warmup_mcmc,
            num_samples_mcmc = num_samples_mcmc,
            num_chains_mcmc = num_chains_mcmc,
            verbose = verbose
    )
    # truncate the FPR output to only the forecast, to be consistent with the other forecasts
    forecast_ix = np.where(pd.to_datetime(forecast_fpr_logistic_date_vec) > pd.to_datetime(train_end_date))[0][0]
    filplus_rate_logistic_pred = np.asarray(filplus_rate_logistic_pred)[:, forecast_ix:]
    forecast_fpr_logistic_date_vec = forecast_fpr_logistic_date_vec[forecast_ix:]

    # split + combine the FIL+ forecasts
    filplus_rate_pred_spliced = np.zeros_like(filplus_rate_pred)
    total_nummc = filplus_rate_pred.shape[0]
    num_sgt = int(total_nummc*3./4.)
    num_logistic = total_nummc - num_sgt
    filplus_rate_pred_spliced[:num_sgt, :] = filplus_rate_pred[:num_sgt, :]
    filplus_rate_pred_spliced[num_logistic:, :] = filplus_rate_logistic_pred[num_logistic:, :]
    filplus_rate_pred_spliced = jnp.copy(jnp.asarray(filplus_rate_pred_spliced))
    ##########################################################################################

    # # debugging
    # import pickle
    # with open('/tmp/debug.pkl', 'wb') as f:
    #     pickle.dump({
    #         'filplus_rate_pred_sgt': np.asarray(filplus_rate_pred),
    #         'filplus_rate_pred_logistic': np.asarray(filplus_rate_logistic_pred),
    #         'filplus_rate_pred_spliced': np.asarray(filplus_rate_pred_spliced),
    #         'num_sgt': num_sgt,
    #         'num_logistic': num_logistic,
    #     }, f)
    
    diagnostics = {
        'rb_rhats': rb_rhats,
        'ext_rhats': ext_rhats,
        'expire_rhats': expire_rhats,
        'deal_onboard_rhats': deal_onboard_pred_rhats,
        'cc_onboard_rhats': cc_onboard_pred_rhats,
        'fpr_rhat': fpr_rhat,
    }
    
    return rb_onboard_power_pred, renewal_rate_pred, filplus_rate_pred_spliced, historical_rb_date, historical_rb, historical_rr_date, historical_rr, historical_fpr_date, historical_fpr, diagnostics
    
def get_fil_historical_price(history_n_days=90):
    fil = yf.Ticker('FIL-USD')
    price = fil.history(period='max')
    # filter to last N days
    price = price.iloc[-history_n_days:]
    return price['Close'].median()

def run_mcmc(
    mcmc_train_start_date,
    mcmc_train_end_date,
    forecast_length,
    num_warmup_mcmc = 10000,
    num_samples_mcmc = 500,
    seasonality_mcmc = 2000,
    num_chains_mcmc = 4,
):
    rb_onboard_power_pred, renewal_rate_pred, filplus_rate_pred, historical_rb_date, historical_rb, historical_rr_date, historical_rr, historical_fpr_date, historical_fpr, diagnostics = \
        generate_mcmc_forecast_samples(mcmc_train_start_date,
                                       mcmc_train_end_date,
                                       forecast_length,
                                       num_warmup_mcmc,
                                       num_samples_mcmc,
                                       seasonality_mcmc,
                                       num_chains_mcmc,
                                       verbose=True)
    return {
        'rb_onboard_power_pred': rb_onboard_power_pred,
        'renewal_rate_pred': renewal_rate_pred,
        'filplus_rate_pred': filplus_rate_pred,
        'historical_rb_date': historical_rb_date,
        'historical_rb': historical_rb,
        'historical_rr_date': historical_rr_date,
        'historical_rr': historical_rr,
        'historical_fpr_date': historical_fpr_date,
        'historical_fpr': historical_fpr,
        'diagnostics': diagnostics
    }

@memory.cache
def get_onboarding_historical_data(current_date, history_days=360):
    hist_plot_tvec_rr, hist_rr = u.get_historical_renewal_rate(current_date-timedelta(days=history_days), current_date)
    hist_plot_tvec_rbp, hist_rbp = u.get_historical_daily_onboarded_power(current_date-timedelta(days=history_days), current_date)
    hist_plot_tvec_fpr, hist_fpr = u.get_historical_filplus_rate(current_date-timedelta(days=history_days), current_date)
    return {
        'hist_plot_tvec_rr': hist_plot_tvec_rr,
        'hist_rr': hist_rr,
        'hist_plot_tvec_rbp': hist_plot_tvec_rbp,
        'hist_rbp': hist_rbp,
        'hist_plot_tvec_fpr': hist_plot_tvec_fpr,
        'hist_fpr': hist_fpr,
    }

@memory.cache
def get_historical_kpis(start_date, current_date):
    # hist_df = pystarboard.data.get_historical_network_stats(start_date-timedelta(days=360), current_date, end_date)
    # hist_expire_df = pystarboard.data.query_sector_expirations(start_date-timedelta(days=360), current_date)
    # hist_econ_df = pystarboard.data.query_sector_economics(
    #     start_date-timedelta(days=360), 
    #     current_date, 
    # )
    hist_df = pystarboard.data.get_historical_network_stats(start_date, current_date, current_date)
    hist_expire_df = pystarboard.data.query_sector_expirations(start_date, current_date)
    hist_econ_df = pystarboard.data.query_sector_economics(
        start_date, 
        current_date, 
    )
        
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df.sort_values('date', inplace=True)
    hist_expire_df['date'] = pd.to_datetime(hist_expire_df['date']).dt.tz_localize(None)
    hist_expire_df.sort_values('date', inplace=True)
    hist_df_merged = pd.merge_asof(hist_df, hist_expire_df, on='date')
    hist_df_merged['date'] = pd.to_datetime(hist_df_merged['date'])

    pledge_historical = hist_econ_df[['date', 'sector_initial_pledge_32gib']]
    pledge_historical['date'] = pd.to_datetime(pledge_historical['date'])
    
    # pledge_historical.rename(columns={'sector_initial_pledge_32gib': 'day_pledge_per_QAP'}, inplace=True)
    hist_df_merged = pd.merge_asof(hist_df_merged, pledge_historical, on='date')

    hist_df_merged['day_network_reward'] = hist_df_merged['mined_fil'].diff()
    hist_df_merged['day_network_reward'].iloc[0] = hist_df_merged['day_network_reward'].iloc[1]
    hist_df_merged['day_locked_pledge'] = hist_df_merged['locked_fil'].diff()
    hist_df_merged['day_renewed_pledge'] = hist_df_merged['extended_pledge'].diff()

    hist_df_merged['day_rewards_per_sector'] = C.EIB_PER_SECTOR * hist_df_merged['day_network_reward'] / hist_df_merged['total_qa_power_eib']

    # rename to what mechafil expects
    hist_df_merged.rename(
        columns={
            'total_qa_power_eib': 'qa_total_power_eib',
            'sector_initial_pledge_32gib': 'day_pledge_per_QAP',
            'circulating_fil': 'circ_supply',
            'locked_fil': 'network_locked',
        }, 
        inplace=True
    )
    return hist_df_merged

def fix_fp(fp, prefix=None, postfix=None):
    fname, ext = os.path.splitext(fp)
    if prefix is not None and postfix is not None:
        fp = '%s_%s_%s%s' % (prefix, fname, postfix, ext)
    elif prefix is not None and postfix is None:
        fp = '%s_%s%s' % (prefix, fname, ext)
    elif prefix is None and postfix is not None:
        fp = '%s_%s%s' % (fname, postfix, ext)
    else:
        fp = '%s%s' % (fname, ext)
    return fp


def generate_network_mcmc_forecast_plots(
    mcmc_trajectories, 
    mcmc_simulation_results_vec, 
    labels_vec,
    start_date, 
    current_date, 
    end_date, 
    save_dir,
    save_fp_prefix=None,
    save_fp_postfix=None,
    vlines=[],
    vline_labels=[],
    hlines=[],
    hline_labels=[],
):
    os.makedirs(save_dir, exist_ok=True)
    # plot inputs
    hist_inputs = get_onboarding_historical_data(current_date, history_days=180)
    hist_inputs = SimpleNamespace(**hist_inputs)
    fp = fix_fp('inputs.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_inputs(mcmc_trajectories, hist_inputs, current_date, os.path.join(save_dir, fp))

    hist_kpi_df = get_historical_kpis(
        current_date-timedelta(days=180), 
        current_date, 
        # current_date+timedelta(days=365*2)
    )

    fp = fix_fp('power.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_mcmc_power_panel(
        hist_kpi_df, 
        mcmc_simulation_results_vec, 
        labels_vec,
        start_date, 
        current_date, 
        end_date, 
        vlines,
        vline_labels,
        os.path.join(save_dir, fp)
    )

    fp = fix_fp('supply.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_mcmc_supply_panel(
        hist_kpi_df, 
        mcmc_simulation_results_vec, 
        labels_vec,
        start_date, 
        current_date, 
        end_date, 
        vlines,
        vline_labels,
        hlines,
        hline_labels,
        os.path.join(save_dir, fp)
    )

    fp = fix_fp('onboarding.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_mcmc_onboarding_panel(
        hist_kpi_df,
        mcmc_simulation_results_vec,
        labels_vec,
        start_date,
        current_date,
        end_date,
        vlines,
        vline_labels,
        os.path.join(save_dir, fp)
    )

def generate_network_mcmc_forecast_plots_delta(
    mcmc_simulation_results_vec, 
    start_date, 
    current_date, 
    end_date, 
    save_dir,
    save_fp_prefix=None,
    save_fp_postfix=None,
):
    os.makedirs(save_dir, exist_ok=True)

    fp = fix_fp('supply.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_mcmc_supply_panel_delta(
        mcmc_simulation_results_vec, 
        start_date, 
        current_date, 
        end_date, 
        os.path.join(save_dir, fp)
    )

    fp = fix_fp('onboarding.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_mcmc_onboarding_panel_delta(
        mcmc_simulation_results_vec,
        start_date,
        current_date,
        end_date,
        os.path.join(save_dir, fp)
    )

def generate_scenario_forecast_plots(
    simconfig2results,
    rbp_factors,
    rr_factors,
    fpr_factors,
    start_date, 
    current_date, 
    end_date, 
    save_dir,
    save_fp_prefix=None,
    save_fp_postfix=None,
):
    hist_kpi_df = get_historical_kpis(
        current_date-timedelta(days=180), 
        current_date, 
        current_date+timedelta(days=365*2)
    )

    fp = fix_fp('power_scenarios.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_power_scenarios(
        hist_kpi_df,
        simconfig2results,
        start_date,
        current_date,
        end_date,
        rbp_factors,
        rr_factors,
        fpr_factors,
        os.path.join(save_dir, fp)
    )

    fp = fix_fp('supply_scenarios.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_supply_scenarios(
        hist_kpi_df,
        simconfig2results,
        start_date,
        current_date,
        end_date,
        rbp_factors,
        rr_factors,
        fpr_factors,
        os.path.join(save_dir, fp)
    )

    fp = fix_fp('onboarding_scenarios.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_onboarding_scenarios(
        hist_kpi_df,
        simconfig2results,
        start_date,
        current_date,
        end_date,
        rbp_factors,
        rr_factors,
        fpr_factors,
        os.path.join(save_dir, fp)
    )

def get_locked_value_distribution(mcmc_simulation_results_vec, start_date, end_date, target_value_locked_usd=100e6, fil_price=5):
    # get the distribution of when Locked USD value will reach the configured value for the given FIL price
    macro_t = du.get_t(start_date, end_date=end_date)
    dates_vec = []
    for ii in range(len(mcmc_simulation_results_vec)):
        simulation_results = mcmc_simulation_results_vec[ii]
        locked_usd = simulation_results['network_locked'] * fil_price
        ix = np.where(locked_usd < target_value_locked_usd)[0]
        if len(ix) > 0:
            dates_vec.append(macro_t[ix[0]])
            
    return np.asarray(dates_vec)


def get_newpledge_oldpledge_delta_distribution(mcmc_simulation_results_vec, current_date, target_threshold_pct=100):
    # get the distribution of when NewPledge > 2*CurrentPledge
    pass

# TODO: expand the logic
def get_upgrade_date_mcmc(filprice2lvd, fil_price=3, q=0.05):
    # default values are conservative
    upgrade_date = np.quantile(filprice2lvd[fil_price], q)
    return upgrade_date

def get_upgrade_date_scenario(sim2configresults, start_date, end_date, fil_price=3, target_value_locked_usd=100e6):
    # choose the soonest upgrade date from all scenarios computed
    # this is a conservative approach
    min_date = None
    macro_t = du.get_t(start_date, end_date=end_date)
    for sim_config, simulation_results in sim2configresults.items():
        locked_usd = simulation_results['network_locked'] * fil_price
        ix = np.where(locked_usd < target_value_locked_usd)[0]
        if len(ix) > 0:
            if min_date is None or macro_t[ix[0]] < min_date:
                min_date = macro_t[ix[0]]
    return min_date

def create_gamma_vector(upgrade_date, forecast_length, current_date, ramp_len_days=90):
    gamma_smooth = np.ones(forecast_length)
    update_day = (upgrade_date-pd.to_datetime(current_date)).days

    ramp_gamma = np.linspace(1, 0.7, ramp_len_days)
    
    ramp_start_idx = update_day
    ramp_end_idx = min(forecast_length, ramp_start_idx + ramp_len_days)
    gamma_smooth[ramp_start_idx:ramp_end_idx] = ramp_gamma[0:(ramp_end_idx-ramp_start_idx)]
    gamma_smooth[ramp_end_idx:] = 0.7
    return gamma_smooth

def main(
    auth_token=os.path.join(os.environ['HOME'],'code/cel/auth/spacescope_auth.json'), 
    mcmc_train_len_days=90, 
    output_dir='plots',
    forecast_start_date=None,
    num_warmup_mcmc = 10000,
    num_samples_mcmc = 500,
    seasonality_mcmc = 2000,
    num_chains_mcmc = 4,
    rhat_threshold_pct=95,
):
    os.makedirs(output_dir, exist_ok=True)
    lock_target = 0.3
    sector_duration = 365

    if forecast_start_date is None:
        forecast_start_date = date.today() - timedelta(days=3)  # starboard data aggregation delay
    current_date = forecast_start_date  # legacy variable naming that's sort of hard to understand
    start_date = current_date - timedelta(days=3)  # historical data for forecasting.  keep this as short as possible
                                                   # to reduce locking discrepancy
    forecast_length = 365*3
    end_date = current_date + timedelta(days=forecast_length)

    pystarboard.data.setup_spacescope(auth_token)

    ############################################################################
    ### MCMC Forecasting of user inputs
    mcmc_train_end_date = forecast_start_date - timedelta(days=1)
    mcmc_train_start_date = mcmc_train_end_date - timedelta(days=(mcmc_train_len_days))
    mcmc_data = run_mcmc(
        mcmc_train_start_date,
        mcmc_train_end_date,
        forecast_length,
        num_warmup_mcmc,
        num_samples_mcmc,
        seasonality_mcmc,
        num_chains_mcmc
    )

    mcmc_trajectories = SimpleNamespace(**mcmc_data)
    rb_rhat_check = mcmc.check_rhat(mcmc_trajectories.diagnostics['rb_rhats'])*100
    ext_rhat_check = mcmc.check_rhat(mcmc_trajectories.diagnostics['ext_rhats'])*100
    exp_rhat_check = mcmc.check_rhat(mcmc_trajectories.diagnostics['expire_rhats'])*100
    fpr_rhat_check = mcmc.check_rhat(mcmc_trajectories.diagnostics['fpr_rhat'])*100
    deal_onboard_pred_rhats = mcmc.check_rhat(mcmc_trajectories.diagnostics['deal_onboard_rhats'])*100
    cc_onboard_pred_rhats = mcmc.check_rhat(mcmc_trajectories.diagnostics['cc_onboard_rhats'])*100

    print('RBP Forecast RHat < 1.05: %0.02f %%' % rb_rhat_check)
    print('Extensions Forecast RHat < 1.05: %0.02f %%' % ext_rhat_check)
    print('Expirations Forecast RHat < 1.05: %0.02f %%' % exp_rhat_check)
    print('FIL+ Forecast RHat < 1.05: %0.02f %%' % fpr_rhat_check)
    print('Deal Onboarding Forecast RHat < 1.05: %0.02f %%' % deal_onboard_pred_rhats)
    print('CC Onboarding Forecast RHat < 1.05: %0.02f %%' % cc_onboard_pred_rhats)
    rhats = [rb_rhat_check, ext_rhat_check, exp_rhat_check, fpr_rhat_check, deal_onboard_pred_rhats, cc_onboard_pred_rhats]
    if np.any(np.asarray(rhats) < rhat_threshold_pct):
        raise ValueError('RHat check failed, please reconfigure MCMC with more samples or chains')
    ############################################################################

    # run simulation for forecasted input trajectories
    simulation_offline_data = download_simulation_data(auth_token, start_date, current_date, end_date)

    # mcmc_results_vec = []
    # for ii in tqdm(range(num_samples_mcmc*num_chains_mcmc)):
    #     rbp_vec = mcmc_trajectories.rb_onboard_power_pred[ii,:]
    #     rr_vec = mcmc_trajectories.renewal_rate_pred[ii,:]
    #     fpr_vec = mcmc_trajectories.filplus_rate_pred[ii,:]
    #     simulation_results = sim.run_sim(
    #         rbp_vec,
    #         rr_vec,
    #         fpr_vec,
    #         lock_target,
        
    #         start_date,
    #         current_date,
    #         forecast_length,
    #         sector_duration,
    #         simulation_offline_data
    #     )
    #     mcmc_results_vec.append(simulation_results)

    # # plots
    # generate_network_mcmc_forecast_plots(
    #     mcmc_trajectories, 
    #     [mcmc_results_vec], 
    #     None,
    #     start_date, 
    #     current_date, 
    #     end_date, 
    #     output_dir,
    # )

    # # ################
    # # # create scenarios for rbp/rr/fpr trajectories to see how they compare w/ MCMC for determining
    # # # when the upgrade date should be
    # # hist_inputs = get_onboarding_historical_data(current_date, history_days=30)  # use last 30 days as the anchor, and consider scenarios +/- 20% from that level
    # # hist_inputs = SimpleNamespace(**hist_inputs)
    # # hist_median_rbp = np.median(hist_inputs.hist_rbp)
    # # hist_median_rr = np.median(hist_inputs.hist_rr)
    # # hist_median_fpr = np.median(hist_inputs.hist_fpr)

    # # rbp_factors = [0.8, 1.2]
    # # rr_factors = [0.8, 1.2]
    # # fpr_factors = [0.8, 1.2]

    # # sim_configs = list(itertools.product(rbp_factors, rr_factors, fpr_factors))
    # # sim_configs.insert(0, (1,1,1))
    # # simconfig2results = {}
    # # for sim_config in sim_configs:
    # #     rbp_factor, rr_factor, fpr_factor = sim_config
    # #     rbp_vec = jnp.ones(forecast_length) * hist_median_rbp * rbp_factor
    # #     rr_vec = jnp.ones(forecast_length) * min(0.99, hist_median_rr * rr_factor)
    # #     fpr_vec = jnp.ones(forecast_length) * min(0.99, hist_median_fpr * fpr_factor)

    # #     simulation_results = sim.run_sim(
    # #         rbp_vec,
    # #         rr_vec,
    # #         fpr_vec,
    # #         lock_target,
        
    # #         start_date,
    # #         current_date,
    # #         forecast_length,
    # #         sector_duration,
    # #         simulation_offline_data
    # #     )
    # #     simconfig2results[sim_config] = simulation_results
    # # generate_scenario_forecast_plots(
    # #     simconfig2results,
    # #     rbp_factors,
    # #     rr_factors,
    # #     fpr_factors,
    # #     start_date, 
    # #     current_date, 
    # #     end_date, 
    # #     output_dir,
    # #     save_fp_prefix=None,
    # #     save_fp_postfix=None,
    # # )
    # # ################

    # # check when the upgrade date should be, based on Locked criteria
    # #  Should be when 5th percentile Locked is expected to reach 100M USD - 6 months
    # historical_median_price = get_fil_historical_price(history_n_days=90)
    # target_value_locked_usd = 100e6
    # fil_price_vec = [historical_median_price, 3, 5, 10]
    # filprice2lvd = {}
    # for fil_price in fil_price_vec:
    #     lvd = get_locked_value_distribution(mcmc_results_vec, start_date, end_date, 
    #                                         target_value_locked_usd=target_value_locked_usd, 
    #                                         fil_price=fil_price)
    #     filprice2lvd[fil_price] = lvd
    # pu.plot_locked_value_distribution(filprice2lvd, target_value_locked_usd, os.path.join(output_dir, 'lvd.png'))

    ###########################################################################################
    # check when the upgrade date should be, based on the pledge criteria
    #  Should be the earliest date where NewPledge > 2*CurrentPledge
    # TODO: include scenarios here
    # upgrade_fil_price = historical_median_price
    # upgrade_date_mcmc_q05 = get_upgrade_date_mcmc(filprice2lvd, fil_price=upgrade_fil_price, q=0.05)
    # upgrade_date_mcmc_q20 = get_upgrade_date_mcmc(filprice2lvd, fil_price=upgrade_fil_price, q=0.20)
    # conservative_upgrade_date_scenarios = get_upgrade_date_scenario(
    #     simconfig2results, 
    #     start_date, 
    #     end_date, 
    #     fil_price=upgrade_fil_price, 
    #     target_value_locked_usd=target_value_locked_usd
    # )
    # upgrade_date = min(upgrade_date_mcmc, conservative_upgrade_date_scenarios)
    upgrade_date = pd.to_datetime('2024-03-01')
    # print('Upgrade Date: [ Q05=%s // Q20=%s ]' % (upgrade_date_mcmc_q05, upgrade_date_mcmc_q20))
    ###########################################################################################

    # for the chosen date, show the forecast of the network after the upgrade w/ the new pledge
    # and the old pledge
    FIL_price = 5.00
    
    # ramp_lens_yrs = [0, 1, 2, 3, 5]
    # factors_vec = [1, 1.01, 1.03, 1.05, 1.07]
    ramp_lens_yrs = [0, 2.5 , 5]
    factors_vec = [1, 1.04, 1.07]
    investments_needed = []
    ramplen_mcmc = []
    for ramp_len_yr, factor in zip(ramp_lens_yrs, factors_vec):
        gamma_smooth_vec = create_gamma_vector(upgrade_date, forecast_length, current_date, ramp_len_days=int(ramp_len_yr*365))
        mcmc_results_gamma_vec = []
        investment_needed_vec = []
        for ii in tqdm(range(num_samples_mcmc*num_chains_mcmc)):
            rbp_vec = mcmc_trajectories.rb_onboard_power_pred[ii,:]*factor
            rr_vec = jnp.clip(mcmc_trajectories.renewal_rate_pred[ii,:]*factor, a_max=0.99)
            fpr_vec = mcmc_trajectories.filplus_rate_pred[ii,:]
            simulation_results = sim.run_sim(
                rbp_vec,
                rr_vec,
                fpr_vec,
                lock_target,
            
                start_date,
                current_date,
                forecast_length,
                sector_duration,
                simulation_offline_data,
                gamma=gamma_smooth_vec,
                gamma_weight_type=0  # means arithmetic weighting
            )
            mcmc_results_gamma_vec.append(simulation_results)

            y_val_fil_new = simulation_results['day_locked_pledge'] - simulation_results['day_renewed_pledge']
            y_val_musd = (y_val_fil_new*FIL_price)/1e6
            ix_start = (current_date-start_date).days
            investment_needed_vec.append(np.sum(y_val_musd[ix_start:]))
            
        investments_needed.append(np.quantile(investment_needed_vec, q=[0.05, 0.25, 0.5, 0.75, 0.95]))
        ramplen_mcmc.append(mcmc_results_gamma_vec)

    vlines = [upgrade_date]
    vline_labels = ['Upgrade Date']
    generate_network_mcmc_forecast_plots(
        mcmc_trajectories, 
        ramplen_mcmc, 
        ['%0.1fY/%0.2fx' % (x,y) for x,y in zip(ramp_lens_yrs, factors_vec)],
        start_date, 
        current_date, 
        end_date, 
        output_dir,
        vlines=vlines,
        vline_labels=vline_labels,
        hlines=[100/3, 100/5, 100/10],
        hline_labels=['$100M-TVL@$3/FIL', '$100M-TVL@$5/FIL', '$100M-TVL@$10/FIL'],
        save_fp_prefix='gamma_breakeven_locked',
    )
    print('*** Break Even Locked ***')
    for (ramp_len, factor, invest_needed) in zip(ramp_lens_yrs, factors_vec, investments_needed):
        print('Ramp Len: %0.01fY // Factor: %0.2f // Investment Needed: %s' % (ramp_len, factor, invest_needed))

    ################################################################################################
    historical_median_price = get_fil_historical_price(history_n_days=90)
    fil_price_vec = [historical_median_price, 3]
    # target_value_locked_usd = 100e6

    ramp_lens_yrs = [0, 2.5 , 5]
    factors_vec = [1, 1, 1]
    investments_needed = []
    ramplen_mcmc = []
    for ramp_len_yr, factor in zip(ramp_lens_yrs, factors_vec):
        gamma_smooth_vec = create_gamma_vector(upgrade_date, forecast_length, current_date, ramp_len_days=int(ramp_len_yr*365))
        mcmc_results_gamma_vec = []
        investment_needed_vec = []
        for ii in tqdm(range(num_samples_mcmc*num_chains_mcmc)):
            rbp_vec = mcmc_trajectories.rb_onboard_power_pred[ii,:]*factor
            rr_vec = jnp.clip(mcmc_trajectories.renewal_rate_pred[ii,:]*factor, a_max=0.99)
            fpr_vec = mcmc_trajectories.filplus_rate_pred[ii,:]
            simulation_results = sim.run_sim(
                rbp_vec,
                rr_vec,
                fpr_vec,
                lock_target,
            
                start_date,
                current_date,
                forecast_length,
                sector_duration,
                simulation_offline_data,
                gamma=gamma_smooth_vec,
                gamma_weight_type=0  # means arithmetic weighting
            )
            mcmc_results_gamma_vec.append(simulation_results)

            y_val_fil_new = simulation_results['day_locked_pledge'] - simulation_results['day_renewed_pledge']
            y_val_musd = (y_val_fil_new*FIL_price)/1e6
            ix_start = (current_date-start_date).days
            investment_needed_vec.append(np.sum(y_val_musd[ix_start:]))
            
        investments_needed.append(np.quantile(investment_needed_vec, q=[0.05, 0.25, 0.5, 0.75, 0.95]))
        ramplen_mcmc.append(mcmc_results_gamma_vec)

        filprice2lvd = {}
        for fil_price in fil_price_vec:
            lvd = get_locked_value_distribution(mcmc_results_gamma_vec, start_date, end_date, 
                                                target_value_locked_usd=100e6, 
                                                fil_price=fil_price)
            filprice2lvd[fil_price] = lvd
        pu.plot_locked_value_distribution(filprice2lvd, 100e6, os.path.join(output_dir, 'lvd_noincr_100M_ramp%0.02f.png' % ramp_len_yr))
        
        filprice2lvd = {}
        for fil_price in fil_price_vec:
            lvd = get_locked_value_distribution(mcmc_results_gamma_vec, start_date, end_date, 
                                                target_value_locked_usd=200e6, 
                                                fil_price=fil_price)
            filprice2lvd[fil_price] = lvd
        pu.plot_locked_value_distribution(filprice2lvd, 200e6, os.path.join(output_dir, 'lvd_noincr_200M_ramp%0.02f.png' % ramp_len_yr))

    vlines = [upgrade_date]
    vline_labels = ['Upgrade Date']
    generate_network_mcmc_forecast_plots(
        mcmc_trajectories, 
        ramplen_mcmc, 
        ['%0.1fY/%0.2fx' % (x,y) for x,y in zip(ramp_lens_yrs, factors_vec)],
        start_date, 
        current_date, 
        end_date, 
        output_dir,
        vlines=vlines,
        vline_labels=vline_labels,
        hlines=[100/3, 100/5, 100/10],
        hline_labels=['$100M-TVL@$3/FIL', '$100M-TVL@$5/FIL', '$100M-TVL@$10/FIL'],
        save_fp_prefix='gamma_noincr',
    )
    print('*** No Increase ***')
    for (ramp_len, factor, invest_needed) in zip(ramp_lens_yrs, factors_vec, investments_needed):
        print('Ramp Len: %0.01fY // Factor: %0.2f // Investment Needed: %s' % (ramp_len, factor, invest_needed))
    ################################################################################################
        

    # #### Gamma Inverse
    # ramp_lens_yrs = [0, 2.5, 5]
    # factors_vec = [0.91, 0.95, 1]
    # investments_needed = []
    # ramplen_mcmc = []
    # for ramp_len_yr, factor in zip(ramp_lens_yrs, factors_vec):
    #     gamma_smooth_vec = create_gamma_vector(upgrade_date, forecast_length, current_date, ramp_len_days=int(ramp_len_yr*365))
    #     mcmc_results_gamma_vec = []
    #     investment_needed_vec = []
    #     for ii in tqdm(range(num_samples_mcmc*num_chains_mcmc)):
    #         rbp_vec = mcmc_trajectories.rb_onboard_power_pred[ii,:]*factor
    #         rr_vec = jnp.clip(mcmc_trajectories.renewal_rate_pred[ii,:]*factor, a_max=0.99)
    #         fpr_vec = mcmc_trajectories.filplus_rate_pred[ii,:]
    #         simulation_results = sim.run_sim(
    #             rbp_vec,
    #             rr_vec,
    #             fpr_vec,
    #             lock_target,
            
    #             start_date,
    #             current_date,
    #             forecast_length,
    #             sector_duration,
    #             simulation_offline_data,
    #             gamma=gamma_smooth_vec,
    #             gamma_weight_type=0  # means arithmetic weighting
    #         )
    #         mcmc_results_gamma_vec.append(simulation_results)

    #         y_val_fil_new = simulation_results['day_locked_pledge'] - simulation_results['day_renewed_pledge']
    #         y_val_musd = (y_val_fil_new*FIL_price)/1e6
    #         ix_start = (current_date-start_date).days
    #         investment_needed_vec.append(np.sum(y_val_musd[ix_start:]))
            
    #     investments_needed.append(np.quantile(investment_needed_vec, q=[0.05, 0.25, 0.5, 0.75, 0.95]))
    #     ramplen_mcmc.append(mcmc_results_gamma_vec)

    # vlines = [upgrade_date]
    # vline_labels = ['Upgrade Date']
    # generate_network_mcmc_forecast_plots(
    #     mcmc_trajectories, 
    #     ramplen_mcmc, 
    #     ['%0.1fY/%0.2fx' % (x,y) for x,y in zip(ramp_lens_yrs, factors_vec)],
    #     start_date, 
    #     current_date, 
    #     end_date, 
    #     output_dir,
    #     vlines=vlines,
    #     vline_labels=vline_labels,
    #     hlines=[100/3, 100/5, 100/10],
    #     hline_labels=['$100M-TVL@$3/FIL', '$100M-TVL@$5/FIL', '$100M-TVL@$10/FIL'],
    #     save_fp_prefix='gamma_inverse',
    # )
    # print('*** Inverse ***')
    # for (ramp_len, factor, invest_needed) in zip(ramp_lens_yrs, factors_vec, investments_needed):
    #     print('Ramp Len: %0.1fY // Factor: %0.2f // Investment Needed: %s' % (ramp_len, factor, invest_needed))

    #### Now see, with the same investment, what factor increase would you get w/ a ramped upgrade
    # ramp_lens_yrs = [0, 1, 2, 3, 5]
    # factors_vec = [1, 1.07, 1.15, 1.2, 1.3]
    # investments_needed = []
    # ramplen_mcmc = []
    # for ramp_len_yr, factor in zip(ramp_lens_yrs, factors_vec):
    #     gamma_smooth_vec = create_gamma_vector(upgrade_date, forecast_length, current_date, ramp_len_days=int(ramp_len_yr*365))
    #     mcmc_results_gamma_vec = []
    #     investment_needed_vec = []
    #     for ii in tqdm(range(num_samples_mcmc*num_chains_mcmc)):
    #         rbp_vec = mcmc_trajectories.rb_onboard_power_pred[ii,:]*factor
    #         rr_vec = jnp.clip(mcmc_trajectories.renewal_rate_pred[ii,:]*factor, a_max=0.99)
    #         fpr_vec = mcmc_trajectories.filplus_rate_pred[ii,:]
    #         simulation_results = sim.run_sim(
    #             rbp_vec,
    #             rr_vec,
    #             fpr_vec,
    #             lock_target,
            
    #             start_date,
    #             current_date,
    #             forecast_length,
    #             sector_duration,
    #             simulation_offline_data,
    #             gamma=gamma_smooth_vec,
    #             gamma_weight_type=0  # means arithmetic weighting
    #         )
    #         mcmc_results_gamma_vec.append(simulation_results)

    #         y_val_fil_new = simulation_results['day_locked_pledge'] - simulation_results['day_renewed_pledge']
    #         y_val_musd = (y_val_fil_new*FIL_price)/1e6
    #         ix_start = (current_date-start_date).days
    #         investment_needed_vec.append(np.sum(y_val_musd[ix_start:]))
            
    #     investments_needed.append(np.quantile(investment_needed_vec, q=[0.05, 0.25, 0.5, 0.75, 0.95]))
    #     ramplen_mcmc.append(mcmc_results_gamma_vec)

    # vlines = [upgrade_date]
    # vline_labels = ['Upgrade Date']
    # generate_network_mcmc_forecast_plots(
    #     mcmc_trajectories, 
    #     ramplen_mcmc, 
    #     ['%0.2fx/%dY' % (x,y) for x,y in zip(ramp_lens_yrs, factors_vec)],
    #     start_date, 
    #     current_date, 
    #     end_date, 
    #     output_dir,
    #     vlines=vlines,
    #     vline_labels=vline_labels,
    #     hlines=[100/3, 100/5, 100/10],
    #     hline_labels=['$100M-TVL@$3/FIL', '$100M-TVL@$5/FIL', '$100M-TVL@$10/FIL'],
    #     save_fp_prefix='gamma_breakeven_investment',
    # )
    # print('*** Break Even Investment ***')
    # for (ramp_len, factor, invest_needed) in zip(ramp_lens_yrs, factors_vec, investments_needed):
    #     print('Ramp Len: %dY // Factor: %0.2f // Investment Needed: %s' % (ramp_len, factor, invest_needed))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline Bug Monitor')

    parser.add_argument('--mcmc-train-len-days', type=int, default=90)
    parser.add_argument('--plot-dir', type=str, default='plots')
    parser.add_argument('--starboard-token', type=str, required=True)
    parser.add_argument('--forecast-start-date', type=str, default=None)
    parser.add_argument('--num-warmup-mcmc', type=int, default=10000)
    parser.add_argument('--num-samples-mcmc', type=int, default=500)
    parser.add_argument('--seasonality-mcmc', type=int, default=2000)
    parser.add_argument('--num-chains-mcmc', type=int, default=4)
    args = parser.parse_args()

    main(
        auth_token=args.starboard_token,
        mcmc_train_len_days=args.mcmc_train_len_days,
        output_dir=args.plot_dir,
        forecast_start_date=args.forecast_start_date,
        num_warmup_mcmc=args.num_warmup_mcmc,
        num_samples_mcmc=args.num_samples_mcmc,
        seasonality_mcmc=args.seasonality_mcmc,
        num_chains_mcmc=args.num_chains_mcmc,
    )
