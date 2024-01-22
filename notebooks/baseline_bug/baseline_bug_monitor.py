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
    forecast_rb_date_vec, rb_onboard_power_pred, historical_rb_date, historical_rb, rb_rhats = \
        mcmc.forecast_rb_onboard_power(train_start_date, 
                                       train_end_date,
                                       forecast_length,
                                       num_warmup_mcmc = num_warmup_mcmc,
                                       num_samples_mcmc = num_samples_mcmc,
                                       seasonality_mcmc = seasonality_mcmc,
                                       num_chains_mcmc = num_chains_mcmc,
                                       verbose = verbose)
    
    if verbose: print("Forecasting Renewal Rate")
    forecast_rr_date_vec, renewal_rate_pred, historical_rr_date , historical_rr, ext_rhats, expire_rhats = \
        mcmc.forecast_renewal_rate(train_start_date, 
                                   train_end_date,
                                   forecast_length,
                                   num_warmup_mcmc = num_warmup_mcmc,
                                   num_samples_mcmc = num_samples_mcmc,
                                   seasonality_mcmc = seasonality_mcmc,
                                   num_chains_mcmc = num_chains_mcmc,
                                   verbose = verbose)
    
    if verbose: print("Forecasting FIL+ Rate")
    forecast_fpr_date_vec, filplus_rate_pred, historical_fpr_date, historical_fpr, fpr_rhat = \
        mcmc.forecast_filplus_rate_logistic(
            train_end_date,
            forecast_length,
            num_warmup_mcmc = num_warmup_mcmc,
            num_samples_mcmc = num_samples_mcmc,
            num_chains_mcmc = num_chains_mcmc,
            verbose = verbose
    )
    # truncate the FPR output to only the forecast, to be consistent with the other forecasts
    forecast_ix = np.where(pd.to_datetime(forecast_fpr_date_vec) > pd.to_datetime(train_end_date))[0][0]
    filplus_rate_pred = np.asarray(filplus_rate_pred)[:, forecast_ix:]
    forecast_fpr_date_vec = forecast_fpr_date_vec[forecast_ix:]
    
    diagnostics = {
        'rb_rhats': rb_rhats,
        'ext_rhats': ext_rhats,
        'expire_rhats': expire_rhats,
        'fpr_rhat': fpr_rhat,
    }
    
    return rb_onboard_power_pred, renewal_rate_pred, filplus_rate_pred, historical_rb_date, historical_rb, historical_rr_date, historical_rr, historical_fpr_date, historical_fpr, diagnostics
    
@memory.cache
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
def get_onboarding_historical_data(current_date):
    hist_plot_tvec_rr, hist_rr = u.get_historical_renewal_rate(current_date-timedelta(days=360), current_date)
    hist_plot_tvec_rbp, hist_rbp = u.get_historical_daily_onboarded_power(current_date-timedelta(days=360), current_date)
    hist_plot_tvec_fpr, hist_fpr = u.get_historical_filplus_rate(current_date-timedelta(days=360), current_date)
    return {
        'hist_plot_tvec_rr': hist_plot_tvec_rr,
        'hist_rr': hist_rr,
        'hist_plot_tvec_rbp': hist_plot_tvec_rbp,
        'hist_rbp': hist_rbp,
        'hist_plot_tvec_fpr': hist_plot_tvec_fpr,
        'hist_fpr': hist_fpr,
    }

@memory.cache
def get_historical_kpis(start_date, current_date, end_date):
    hist_df = pystarboard.data.get_historical_network_stats(start_date-timedelta(days=360), current_date, end_date)
    hist_expire_df = pystarboard.data.query_sector_expirations(start_date-timedelta(days=360), current_date)
    hist_econ_df = pystarboard.data.query_sector_economics(
        start_date-timedelta(days=360), 
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
    start_date, 
    current_date, 
    end_date, 
    save_dir,
    save_fp_prefix=None,
    save_fp_postfix=None,
):
    os.makedirs(save_dir, exist_ok=True)
    # plot inputs
    hist_inputs = get_onboarding_historical_data(current_date)
    hist_inputs = SimpleNamespace(**hist_inputs)
    fp = fix_fp('inputs.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_inputs(mcmc_trajectories, hist_inputs, current_date, os.path.join(save_dir, fp))

    hist_kpi_df = get_historical_kpis(
        current_date-timedelta(days=180), 
        current_date, 
        current_date+timedelta(days=365*2)
    )

    fp = fix_fp('power.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_mcmc_power_panel(
        hist_kpi_df, 
        mcmc_simulation_results_vec, 
        start_date, 
        current_date, 
        end_date, 
        os.path.join(save_dir, fp)
    )

    fp = fix_fp('supply.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_supply_panel(
        hist_kpi_df, 
        mcmc_simulation_results_vec, 
        start_date, 
        current_date, 
        end_date, 
        os.path.join(save_dir, fp)
    )

    fp = fix_fp('onboarding.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_onboarding_panel(
        hist_kpi_df,
        mcmc_simulation_results_vec,
        start_date,
        current_date,
        end_date,
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
    pu.plot_supply_panel_delta(
        mcmc_simulation_results_vec, 
        start_date, 
        current_date, 
        end_date, 
        os.path.join(save_dir, fp)
    )

    fp = fix_fp('onboarding.png', prefix=save_fp_prefix, postfix=save_fp_postfix)
    pu.plot_onboarding_panel_delta(
        mcmc_simulation_results_vec,
        start_date,
        current_date,
        end_date,
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
def get_upgrade_date(filprice2lvd, fil_price=3, q=0.05):
    # default values are conservative
    upgrade_date = np.quantile(filprice2lvd[fil_price], q)
    return upgrade_date

def create_gamma_vector(upgrade_date, forecast_length, current_date, ramp_len_days=90):
    gamma_smooth = np.ones(forecast_length)
    update_day_end = pd.to_datetime(upgrade_date)
    update_day_start = update_day_end - timedelta(days=ramp_len_days)
    ramp_gamma = np.linspace(1, 0.7, (update_day_end-update_day_start).days)
    ramp_start_idx = (update_day_start-pd.to_datetime(current_date)).days
    ramp_end_idx = (update_day_end-pd.to_datetime(current_date)).days
    gamma_smooth[ramp_start_idx:ramp_end_idx] = ramp_gamma
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

    if forecast_start_date is None:
        forecast_start_date = date.today() - timedelta(days=3)  # starboard data aggregation delay
    current_date = forecast_start_date  # legacy variable naming that's sort of hard to understand
    start_date = current_date - timedelta(days=3)  # historical data for forecasting.  keep this as short as possible
                                                   # to reduce locking discrepancy
    forecast_length = 365*2  # forecast upto 2 yrs in the future
    end_date = current_date + timedelta(days=forecast_length)

    pystarboard.data.setup_spacescope(auth_token)

    ############################################################################
    ### MCMC Forecasting of user inputs
    mcmc_train_start_date = current_date - timedelta(days=(mcmc_train_len_days))
    mcmc_train_end_date = mcmc_train_start_date + timedelta(days=mcmc_train_len_days)
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
    # dealonboard_rhat_check = mcmc.check_rhat(mcmc_trajectories.diagnostics['deal_onboard_pred_rhats'])*100
    # cconboard_rhat_check = mcmc.check_rhat(mcmc_trajectories.diagnostics['cc_onboard_pred_rhats'])*100
    fpr_rhat_check = mcmc.check_rhat(mcmc_trajectories.diagnostics['fpr_rhat'])*100

    print('RBP Forecast RHat < 1.05: %0.02f %%' % rb_rhat_check)
    print('Extensions Forecast RHat < 1.05: %0.02f %%' % ext_rhat_check)
    print('Expirations Forecast RHat < 1.05: %0.02f %%' % exp_rhat_check)
    print('FIL+ Forecast RHat < 1.05: %0.02f %%' % fpr_rhat_check)
    # print('Deal Onboard Forecast RHat < 1.05: %0.02f %%' % dealonboard_rhat_check)
    # print('CC Onboard Forecast RHat < 1.05: %0.02f %%' % cconboard_rhat_check)
    # rhats = [rb_rhat_check, ext_rhat_check, exp_rhat_check, dealonboard_rhat_check, cconboard_rhat_check]
    rhats = [rb_rhat_check, ext_rhat_check, exp_rhat_check, fpr_rhat_check]
    if np.any(np.asarray(rhats) < rhat_threshold_pct):
        raise ValueError('RHat check failed, please reconfigure MCMC with more samples or chains')
    ############################################################################

    # run simulation for forecasted input trajectories
    lock_target = 0.3
    sector_duration = 365
    simulation_offline_data = download_simulation_data(auth_token, start_date, current_date, end_date)

    mcmc_results_vec = []
    for ii in tqdm(range(num_samples_mcmc*num_chains_mcmc)):
        rbp_vec = mcmc_trajectories.rb_onboard_power_pred[ii,:]
        rr_vec = mcmc_trajectories.renewal_rate_pred[ii,:]
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
            simulation_offline_data
        )
        mcmc_results_vec.append(simulation_results)

    # plots
    # TODO: plot vlines
    generate_network_mcmc_forecast_plots(
        mcmc_trajectories, 
        [mcmc_results_vec], 
        start_date, 
        current_date, 
        end_date, 
        output_dir,
    )

    # check when the upgrade date should be, based on Locked criteria
    #  Should be when 5th percentile Locked is expected to reach 100M USD - 6 months
    target_value_locked_usd = 100e6
    fil_price_vec = [3, 5, 10]
    filprice2lvd = {}
    for fil_price in fil_price_vec:
        lvd = get_locked_value_distribution(mcmc_results_vec, start_date, end_date, 
                                            target_value_locked_usd=target_value_locked_usd, 
                                            fil_price=fil_price)
        filprice2lvd[fil_price] = lvd
    pu.plot_locked_value_distribution(filprice2lvd, target_value_locked_usd, os.path.join(output_dir, 'lvd.png'))

    # check when the upgrade date should be, based on the pledge criteria
    #  Should be the earliest date where NewPledge > 2*CurrentPledge

    upgrade_date = get_upgrade_date(filprice2lvd, fil_price=3, q=0.05)

    # for the chosen date, show the forecast of the network after the upgrade w/ the new pledge
    # and the old pledge
    gamma_smooth_vec = create_gamma_vector(upgrade_date, forecast_length, current_date, ramp_len_days=90)
    mcmc_results_gamma_vec = []
    for ii in tqdm(range(num_samples_mcmc*num_chains_mcmc)):
        rbp_vec = mcmc_trajectories.rb_onboard_power_pred[ii,:]
        rr_vec = mcmc_trajectories.renewal_rate_pred[ii,:]
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
    generate_network_mcmc_forecast_plots(
        mcmc_trajectories, 
        [mcmc_results_gamma_vec], 
        start_date, 
        current_date, 
        end_date, 
        output_dir,
        save_fp_prefix='gamma',
    )

    # plot overlay
    generate_network_mcmc_forecast_plots(
        mcmc_trajectories, 
        [mcmc_results_vec, mcmc_results_gamma_vec], 
        start_date, 
        current_date, 
        end_date, 
        output_dir,
        save_fp_prefix='overlay',
    )

    # show the delta
    keys = [
        'rb_total_power_eib', 'qa_total_power_eib', 'day_network_reward', 
        'network_locked', 'circ_supply',
        'day_pledge_per_QAP', 'day_rewards_per_sector'
    ]
    delta_vec = []
    for ii in range(num_samples_mcmc*num_chains_mcmc):
        delta = {}
        has_nan = False
        for k in keys:
            y = 100*np.asarray((mcmc_results_gamma_vec[ii][k] - mcmc_results_vec[ii][k])/mcmc_results_vec[ii][k])
            if np.isnan(y).any():
                has_nan = True
                break
            delta[k] = y
        if not has_nan:
            delta_vec.append(delta)
    print(len(delta_vec))
    generate_network_mcmc_forecast_plots_delta(
        delta_vec, 
        start_date, 
        current_date, 
        end_date, 
        output_dir,
        save_fp_prefix='delta',
    )


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
