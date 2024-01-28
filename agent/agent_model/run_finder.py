
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import mlflow


uri = os.environ['MLFLOW_TRACKING_URI']
mlflow.set_tracking_uri(uri)
client = mlflow.tracking.MlflowClient()


class BestRunFinder:
    def __init__(self, metric, experiment, last_n_steps=5,
                 threshold=0.05, tolerance=1, stable_only=True):
        self.metric = metric
        self.experiment = experiment
        self.last_n_steps = last_n_steps
        self.threshold = threshold
        self.tolerance = tolerance
        self.stable_only = stable_only
        self.param_names = ['lr', 'train_batch_size', 'grad_clip', 'explore',
                            'rollout_fragment_length', 'seed', 'n_step', 'adam_epsilon',
                            'target_network_update_freq', 'replay_buffer_config/capacity']

        self.runs = mlflow.search_runs(experiment_names=[experiment], output_format='list')

    def stability_check(self, df):
        df.sort_values(by='timestamp', inplace=True)
        df['rolling_val'] = df['value'].rolling(5).std()
        df['prev_val'] = df['rolling_val'].shift(1)
        df['diff_pct'] = abs((df['prev_val'] - df['rolling_val']) / df['prev_val'])
        last_n = df['diff_pct'].tolist()[-self.last_n_steps:]

        first_val = df['value'].iloc[0]
        last_val = df['value'].iloc[-1]
        flat_bools = [l < self.threshold for l in last_n]

        if self.last_n_steps - sum(flat_bools) <= self.tolerance and first_val < last_val:
            return True
        else:
            return False

    def get_best_runs(self):
        best_runs = []
        for r in self.runs:
            if r.data.metrics.get(self.metric, None):
                last_m = r.data.metrics[self.metric]
                m = client.get_metric_history(r.info.run_id, self.metric)
                m_df = pd.DataFrame.from_records([dict(i) for i in m])
                is_stable = self.stability_check(m_df)
                best_runs.append({'experiment': self.experiment,
                                  'metric_name': self.metric,
                                  'run_name': r.data.tags['mlflow.runName'],
                                  'runid': r.info.run_id,
                                  'last_metric': last_m,
                                  'params': r.data.params,
                                  'is_stable': is_stable,
                                  'diffs': m_df['diff_pct'].tolist(),
                                  'steps': m_df['step'].tolist(),
                                  'values': m_df['value'].tolist()})

        return pd.DataFrame.from_records(best_runs)

    @staticmethod
    def get_study_params(b, param_names):
        params = b['params']
        return {k: v for k, v in params.items() if k in param_names}

    def find(self, fig_path=None, topn_best=1):
        best_runs = self.get_best_runs()

        if self.stable_only:
            best_runs = best_runs[best_runs['is_stable']].copy()

        best_runs['rank'] = best_runs['last_metric'].rank(method='dense', ascending=False)

        if not best_runs.empty:
            best = best_runs[best_runs['rank'] <= topn_best].sort_values(by='rank')
            chart_dat = best.set_index('run_name')[['steps', 'values']].apply(pd.Series.explode).reset_index()
            ax = sns.lineplot(chart_dat, x='steps', y='values', hue='run_name')

            if fig_path:
                plt.savefig(os.path.join(fig_path, '../../assets/best_runs.png'), format='png')
            else:
                plt.ion()
                plt.draw()
                plt.pause(0.001)
                input("Press [enter] to continue.")

            records = best.to_dict('records')
            for r in records:
                r['params'] = self.get_study_params(r, param_names=self.param_names)
        else:
            print('No best runs available...try changing parameters')
            records = {}

        return records


if __name__ == '__main__':
    finder = BestRunFinder('WIS-v_gain', 'train-dqn', last_n_steps=8, threshold=0.15, tolerance=2, stable_only=True)
    params = finder.find(fig_path='.', topn_best=5)
    for p in params:
        print(p['run_name'])
        print(p['params'])