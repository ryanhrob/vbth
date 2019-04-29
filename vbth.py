import time
import os
import sys
import re
import argparse
import json
import tempfile
import fnmatch
import multiprocessing
import glob
import tqdm
import numpy as np
import scipy as sp
import pandas as pd
from pandas.compat import OrderedDict


# SPEAKER VERIFIER IMPL ------------------------------------------------------------------------------------------------


# Constants for simulated computation costs of enrol and verify operations. (units: seconds).
enrol_cost = 0.010
verify_per_bvp_cost = 0.00005
verify_independent_cost = 0.00025


class MockSpeakerVerifier:
    def enrol(self, enrol_data):
        """
        Given enrolment data for a user, will analyse the contents and 
        return a "biometric voice print" template for that user.
        """
        self.busy_wait(enrol_cost)
        return {'speaker': enrol_data['speaker'], 'room': enrol_data['room']}

    def verify(self, bvp_list, verify_data):
        """
        Given a list of previously enrolled users' BVPs and some verification 
        data, will comapre the verification to all enrolled users and return a 
        score for each passed in BVP.
        """
        sleep_duration = verify_independent_cost + (verify_per_bvp_cost * len(bvp_list))
        self.busy_wait(sleep_duration)
        return [verify_data[bvp['speaker']][bvp['room']] for bvp in bvp_list]

    def busy_wait(self, seconds):
        """
        Wastes CPU cycles for requested number of seconds to simulate an 
        algorithm usefully using compute.
        """
        end_time = time.perf_counter() + seconds
        while(time.perf_counter() < end_time):
            pass


# ----------------------------------------------------------------------------------------------------------------------


# RUNTEST --------------------------------------------------------------------------------------------------------------


# SpeakerVerifier instance used by the test harness.
sv = MockSpeakerVerifier()


def process_enrol_item(enrol_item):
    """
    Main function run by worker process to consume a single enrol item.
    """
    bvp = sv.enrol(enrol_item['data'])
    return enrol_item['meta']['room'], enrol_item['meta']['speaker'], bvp


def do_process_verify_item(enrol_bvps, enrol_speakers, enrol_rooms, verify_item):
    """
    Main function run by worker process to consume a single verify item.
    """
    scores = sv.verify(enrol_bvps, verify_item['data'])
    data = OrderedDict()
    data['verify_speaker'] = [verify_item['meta']['speaker']] * len(enrol_bvps)
    data['verify_room'] = [verify_item['meta']['room']] * len(enrol_bvps)
    data['enrol_speaker'] = enrol_speakers
    data['enrol_room'] = enrol_rooms
    data['score'] = scores
    return pd.DataFrame(data)


def process_verify_item(args):
    """
    Wrapper function called by Pool infrastructure in worker process. Unwraps arguments and forward calls
    do_process_verify_item() to do the actual work.
    """
    return do_process_verify_item(*args)


def calculate_chunk_size(thread_count, item_count):
    """
    Helper function to determine the optimum number of items to provide to a worker in one package. The bigger the
    number, the less the communication overhead, but it also becomes less parallel. It has been determined empirically
    that it doesn't make much difference above 20 for our usage so clamping as such.
    :param thread_count:    Number of worker threads available.
    :param item_count:      Number of items to process.
    :return:                Calculated chunk size.
    """
    chunk_size = int(item_count / (thread_count * 10))
    if chunk_size < 1:
        chunk_size = 1
    if chunk_size > 20:
        chunk_size = 20
    return chunk_size


def do_runtest_fast(args):
    thread_count = 4
    verify_df = pd.DataFrame()
    enrol_bvps = []
    enrol_speakers = []
    enrol_rooms = []

    with multiprocessing.Pool(thread_count) as p:

        print('\nEnrolling Users...')
        enrol_start_time = time.perf_counter()

        # Collate all the enrolment items.
        enrol_items = [json.load(open(e, 'r')) for e in 
                      glob.glob(os.path.join(args.corpus, '**', 'enrol*.json'), recursive=True)]

        enrol_item_count = len(enrol_items)
        enrol_chunk_size = calculate_chunk_size(thread_count, enrol_item_count)
        e_data = []

        # Run all enrolment items in parallel.
        for d in tqdm.tqdm(p.imap_unordered(process_enrol_item, enrol_items, chunksize=enrol_chunk_size),
                      total=enrol_item_count, desc='Enrolling Users', ascii=True, unit_scale=True):
            e_data.append(d)

        # Unpack all the enrolment results.
        for d in e_data:
            roomname, spkname, bvp = d
            if bvp is not None:
                enrol_rooms.append(roomname)
                enrol_speakers.append(spkname)
                enrol_bvps.append(bvp)

        enrol_run_time = time.perf_counter() - enrol_start_time
        print('Enrolling Users Complete')

        print('\nVerifying Users...')
        verify_start_time = time.perf_counter()

        # Collate all the verification items.
        verify_items = [(enrol_bvps, enrol_speakers, enrol_rooms, json.load(open(v, 'r'))) for v in 
                        glob.glob(os.path.join(args.corpus, '**', 'verify*.json'), recursive=True)]

        verify_item_count = len(verify_items)
        verify_chunk_size = calculate_chunk_size(thread_count, verify_item_count)
        v_data = []

        # Run all verification items in parallel.
        for d in tqdm.tqdm(p.imap_unordered(process_verify_item, verify_items, chunksize=verify_chunk_size),
                      total=verify_item_count, desc='Verifying Users', ascii=True, unit_scale=True):
            v_data.append(d)

        # Unpack all the verification results.
        verify_df = pd.concat(v_data, ignore_index=True)

        verify_run_time = time.perf_counter() - verify_start_time
        print('Verifying Users Complete')

    print('\nSaving Results...')
    verify_df.to_csv(args.csvout)
    print('Saving Results Complete')

    print('\nParameters:')
    print('  enrol_item_count: {}'.format(enrol_item_count))
    print('  verify_item_count: {}'.format(verify_item_count))
    print('  enrol_chunk_size: {}'.format(enrol_chunk_size))
    print('  verify_chunk_size: {}'.format(verify_chunk_size))
    print('  thread_count: {}'.format(thread_count))

    print('\nExecution Times:')
    print('  enrol_run_time: {:.3f} s'.format(enrol_run_time))
    print('  verify_run_time: {:.3f} s'.format(verify_run_time))
    print('  total_run_time: {:.3f} s'.format(enrol_run_time + verify_run_time))


def do_runtest_medium(args):
    verify_df = pd.DataFrame()
    enrol_bvps = []
    enrol_speakers = []
    enrol_rooms = []

    print('\nEnrolling Users...')
    enrol_start_time = time.perf_counter()

    # Collate all the enrolment items.
    enrol_items = [json.load(open(e, 'r')) for e in 
                    glob.glob(os.path.join(args.corpus, '**', 'enrol*.json'), recursive=True)]

    enrol_item_count = len(enrol_items)
    e_data = []

    # Run all enrolment items.
    for d in tqdm.tqdm(map(process_enrol_item, enrol_items),
                    total=enrol_item_count, desc='Enrolling Users', ascii=True, unit_scale=True):
        e_data.append(d)

    # Unpack all the enrolment results.
    for d in e_data:
        roomname, spkname, bvp = d
        if bvp is not None:
            enrol_rooms.append(roomname)
            enrol_speakers.append(spkname)
            enrol_bvps.append(bvp)

    enrol_run_time = time.perf_counter() - enrol_start_time
    print('Enrolling Users Complete')

    print('\nVerifying Users...')
    verify_start_time = time.perf_counter()

    # Collate all the verification items.
    verify_items = [(enrol_bvps, enrol_speakers, enrol_rooms, json.load(open(v, 'r'))) for v in 
                    glob.glob(os.path.join(args.corpus, '**', 'verify*.json'), recursive=True)]

    verify_item_count = len(verify_items)
    v_data = []

    # Run all verification items.
    for d in tqdm.tqdm(map(process_verify_item, verify_items),
                    total=verify_item_count, desc='Verifying Users', ascii=True, unit_scale=True):
        v_data.append(d)

    # Unpack all the verification results.
    verify_df = pd.concat(v_data, ignore_index=True)

    verify_run_time = time.perf_counter() - verify_start_time
    print('Verifying Users Complete')

    print('\nSaving Results...')
    verify_df.to_csv(args.csvout)
    print('Saving Results Complete')

    print('\nParameters:')
    print('  enrol_item_count: {}'.format(enrol_item_count))
    print('  verify_item_count: {}'.format(verify_item_count))

    print('\nExecution Times:')
    print('  enrol_run_time: {:.3f} s'.format(enrol_run_time))
    print('  verify_run_time: {:.3f} s'.format(verify_run_time))
    print('  total_run_time: {:.3f} s'.format(enrol_run_time + verify_run_time))


def do_runtest_slow(args):
    verify_df = pd.DataFrame()

    print('Enroling and Verifying Users...')
    start_time = time.perf_counter()

    # Collate all the enrolment items.
    enrol_items = [json.load(open(e, 'r')) for e in 
                   glob.glob(os.path.join(args.corpus, '**', 'enrol*.json'), recursive=True)]

    # Collate all the verification items.
    verify_items = [json.load(open(v, 'r')) for v in 
                    glob.glob(os.path.join(args.corpus, '**', 'verify*.json'), recursive=True)]

    v_data = []
    for e in tqdm.tqdm(enrol_items,
                       total=len(enrol_items), desc='Enrolling and Verifying Users', ascii=True, unit_scale=True):
        bvp = sv.enrol(e['data'])
        for v in verify_items:
            score = sv.verify([bvp], v['data'])[0]
            data = OrderedDict()
            data['verify_speaker'] = [v['meta']['speaker']]
            data['verify_room'] = [v['meta']['room']]
            data['enrol_speaker'] = [e['meta']['speaker']]
            data['enrol_room'] = [e['meta']['room']]
            data['score'] = [score]
            v_data.append(pd.DataFrame(data))

    # Unpack all the verification results.
    verify_df = pd.concat(v_data, ignore_index=True)

    run_time = time.perf_counter() - start_time
    print('Enrolment and Verification Complete')

    print('\nSaving Results...')
    verify_df.to_csv(args.csvout)
    print('Saving Results Complete')

    print('\nExecution Times:')
    print('  total_run_time: {:.3f} s'.format(run_time))


def do_runtest(args):
    if args.impl == 'fast':
        return do_runtest_fast(args)
    elif args.impl == 'medium':
        return do_runtest_medium(args)
    elif args.impl == 'slow':
        return do_runtest_slow(args)
    else:
        raise ValueError('Unknown impl {}'.format(args.impl))
        

# ----------------------------------------------------------------------------------------------------------------------


# ANALYSE --------------------------------------------------------------------------------------------------------------


def paint_graph(x_label, y_label, df, th_user, label_user, th_calc, label_calc):
    """
    Plots a speaker gram, with target and non-target scores, as well as showing the user threshold and calculated
    threshold. The caller can specify the columns that should be plotted.
    :param x_label:     Column name for the x-axis data. Usually 'score'.
    :param y_label:     Column name for the y-axis data. Allows plotting scores against different variables such as
                        speakers, enrolments, rooms, etc.
    :param df:          Dataframe containing all the data to be plotted.
    :param th_user:     User-defined score threshold.
    :param label_user:  Label describing the User-defined threshold for the legend.
    :param th_calc:     Calculated score threshold.
    :param label_calc:  Label describing the calculated threshold for the legend.
    :return:            None. Saves graph to PNG in current directory and displays in maximised window. Function does
                        not return until window is closed.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = df.copy()
    df['target'] = df['target'].map({True: 'target', False: 'non-target'})
    title = '{} vs {}'.format(x_label, y_label)

    plt.figure(figsize=(1920 / 96, 1103 / 96), dpi=96)

    sns.stripplot(x=x_label, y=y_label, hue='target', data=df,
                  palette={'target': 'blue', 'non-target': 'red'}, alpha=0.5, jitter=False, dodge=True)
    plt.axvline(x=th_user, label=label_user, c='g')
    plt.axvline(x=th_calc, label=label_calc, c='k')
    plt.title(title)
    plt.legend()

    plt.savefig('{}.png'.format(title))

    if plt.get_backend() == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
    plt.show()


def do_analyse(args):
    """
    Analyses a set of results output by runtest and paints various graphs.
    """

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    score = 'score'

    # Read in the results, and add a boolean target column.
    df = pd.read_csv(args.results, index_col=0)
    df['target'] = df['verify_speaker'] == df['enrol_speaker']

    # Calculate ideal 0.01% threshold over the multi-session data.
    nontarget_df = df.loc[df['target'] == False].sort_values(score, ascending=False)
    nontarget_count = nontarget_df[score].count()
    th_calc = nontarget_df.iloc[int(nontarget_count * (1 / 10000))][score]

    # Now filter the data so that we only consider mono-session enrolment and verification.
    df = df.loc[df['verify_room'] == df['enrol_room']]
    target_df = df.loc[df['target'] == True].sort_values(score, ascending=False)
    nontarget_df = df.loc[df['target'] == False].sort_values(score, ascending=False)
    target_count = target_df[score].count()
    nontarget_count = nontarget_df[score].count()

    # Calculate FA/FR for the user-defined threshold.
    th_user = args.th_user
    fr_user = target_df.loc[target_df[score] < th_user][score].count()
    fa_user = nontarget_df.loc[nontarget_df[score] > th_user][score].count()
    frr_user = fr_user / target_count
    far_user = fa_user / nontarget_count
    label_user = 'User Threshold: th {:.4f}, FR {} ({:.3f}%), FA {} ({:.3f}%)'.format(th_user, fr_user, frr_user * 100,
                                                                                      fa_user, far_user * 100)

    # Calculate the FA/FR for the ideal threshold calculated from the multi-session data.
    fr_calc = target_df.loc[target_df[score] < th_calc][score].count()
    fa_calc = nontarget_df.loc[nontarget_df[score] > th_calc][score].count()
    frr_calc = fr_calc / target_count
    far_calc = fa_calc / nontarget_count
    label_calc = 'Calc Threshold: th {:.4f}, FR {} ({:.3f}%), FA {} ({:.3f}%)'.format(th_calc, fr_calc, frr_calc * 100,
                                                                                      fa_calc, far_calc * 100)

    # Print the stats.
    print('\nTarget Stats:')
    print(target_df[score].describe())
    print('\nNon-Target Stats:')
    print(nontarget_df[score].describe())
    print('\nThresholds:')
    print(label_user)
    print(label_calc)

    # Paint the graphs.
    paint_graph(score, 'verify_room', df, th_user, label_user, th_calc, label_calc)
    paint_graph(score, 'enrol_room', df, th_user, label_user, th_calc, label_calc)
    paint_graph(score, 'verify_speaker', df, th_user, label_user, th_calc, label_calc)
    paint_graph(score, 'enrol_speaker', df, th_user, label_user, th_calc, label_calc)


# ----------------------------------------------------------------------------------------------------------------------


# MAIN: COMMAND DISPATCHER ---------------------------------------------------------------------------------------------


def main():
    """
    Main entry point. Parses command line arguments and dispatches to requested command handler.
    """

    # Parse arguments. The parser will raise an exception if required arguments are not present.
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='command')

    # Arguments for the runtest command.
    cmd_runtest = subparsers.add_parser('runtest')
    runtest_required_named = cmd_runtest.add_argument_group('named arguments')
    runtest_required_named.add_argument('-c', '--corpus',
                                    help='Corpus root directory containing all speakers.',
                                    metavar='corpus',
                                    required=True)
    runtest_required_named.add_argument('-o', '--csvout',
                                    help='CSV output file.',
                                    metavar='csvout',
                                    required=True)
    runtest_required_named.add_argument('-i', '--impl',
                                    help='Test runner implementation: fast, medium or slow.',
                                    metavar='impl',
                                    required=False,
                                    default='fastest')

    # Arguments for the analyse command.
    cmd_analyse = subparsers.add_parser('analyse')
    analyse_required_named = cmd_analyse.add_argument_group('named arguments')
    analyse_required_named.add_argument('-r', '--results',
                                    help='Input CSV results file.',
                                    metavar='results',
                                    required=True)
    analyse_required_named.add_argument('-t', '--th_user',
                                    help='User-defined threshold.',
                                    metavar='th_user',
                                    required=False,
                                    type=float,
                                    default=5.79)

    # Parse the arguments.
    args = parser.parse_args()

    # Dispatch to the correct command.
    if args.command == 'runtest':
        do_runtest(args)
    elif args.command == 'analyse':
        do_analyse(args)
    else:
        raise ValueError('Unknown command {}'.format(args.command))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------------------------------------------------
