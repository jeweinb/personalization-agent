
from ray.tune.utils import flatten_dict
from ray.air.callbacks.mlflow import MLflowLoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from agent.agent_model.trainer_utils import clean_json


class CustomMLflowLogger(MLflowLoggerCallback):
    def log_trial_start(self, trial):
        if trial not in self._trial_runs:
            tags = self.tags.copy()
            tags["trial_name"] = str(trial)
            run = self.mlflow_util.start_run(tags=tags, run_name=str(trial))
            self._trial_runs[trial] = run.info.run_id

        run_id = self._trial_runs[trial]
        config = clean_json(trial.config)
        config = flatten_dict(config, delimiter='/')
        self.mlflow_util.log_params(run_id=run_id, params_to_log=config)

    def log_trial_result(self, iteration, trial, result):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        run_id = self._trial_runs[trial]
        is_scores = result['evaluation']['off_policy_estimator']['is']
        is_scores = {f'IS-{k}': v for k, v in is_scores.items() if k in ['v_gain', 'v_target']}
        wis_scores = result['evaluation']['off_policy_estimator']['wis']
        wis_scores = {f'WIS-{k}': v for k, v in wis_scores.items() if k in ['v_gain', 'v_target']}
        dr_scores = result['evaluation']['off_policy_estimator']['dr_fqe']
        dr_scores = {f'DR-{k}': v for k, v in dr_scores.items() if k in ['v_gain', 'v_target']}
        all_scores = {**is_scores, **wis_scores, **dr_scores}
        self.mlflow_util.log_metrics(run_id=run_id, metrics_to_log=all_scores, step=step)
