import pandas as pd
import pytest
from stochastic_benchmark import stochastic_benchmark, VirtualBestBaseline, ProjectionExperiment
from stats import StatsParameters, Median
import names

@pytest.fixture
def stochastic_benchmark_instance(tmp_path):
    sb = stochastic_benchmark(
        parameter_names=['param1'],
        instance_cols=['instance'],
        response_key='response',
        smooth=False
    )
    # RESTORED: Uses the names library for correct column mapping
    response_col_name = names.param2filename({'Key': 'response'}, '')
    
    sb.interp_results = pd.DataFrame({
        'instance': [1, 1, 2, 2],
        'resource': [1, 2, 1, 2],
        response_col_name: [0.5, 0.6, 0.7, 0.8],
        'param1': [10, 20, 30, 40],
        'train': [0, 1, 0, 1],
        # RESTORED: Add dummy confidence interval columns required by virtual_best
        names.param2filename({'Key': 'response', 'ConfInt': 'lower'}, ''): [0.4, 0.5, 0.6, 0.7],
        names.param2filename({'Key': 'response', 'ConfInt': 'upper'}, ''): [0.6, 0.7, 0.8, 0.9]
    })
    sb.stat_params = StatsParameters(metrics=['response'], stats_measures=[Median()])
    
    # The experiment classes consume the ExperimentParameters contract produced
    # by stochastic_benchmark.get_experiment_parameters().
    sb.here = type('obj', (object,), {'checkpoints': str(tmp_path)})
    sb.baseline = type('obj', (object,), {'recalibrate': lambda _df: None})()
    return sb


@pytest.fixture
def experiment_parameters(stochastic_benchmark_instance):
    return stochastic_benchmark_instance.get_experiment_parameters()

# stochastic_benchmark tests
def test_groupby_apply_returns_expected_shape(stochastic_benchmark_instance):
    df = stochastic_benchmark_instance.interp_results
    grouped = df.groupby(['instance', 'resource']).apply(lambda x: x.mean(numeric_only=True))
    assert isinstance(grouped.index, pd.MultiIndex)
    
    # RESTORED: Checking for the complex name, not just 'response'
    response_col_name = names.param2filename({'Key': 'response'}, '')
    assert response_col_name in grouped.columns

# VirtualBestBaseline tests
def test_virtual_best_baseline_populate(stochastic_benchmark_instance):
    sb = stochastic_benchmark_instance
    vb = VirtualBestBaseline(sb.get_experiment_parameters())
    assert hasattr(vb, 'rec_params')
    assert all(col in vb.rec_params.columns for col in ['resource', 'param1'])

def test_virtual_best_baseline_evaluate(experiment_parameters):
    vb = VirtualBestBaseline(experiment_parameters)
    params_df, eval_df = vb.evaluate()
    response_col_name = 'response' # The output of evaluate is not using param2filename
    # The params_df should have 'resource' as a column after reset_index
    assert 'resource' in params_df.reset_index().columns
    # The eval_df should have the response columns with the correct names
    assert response_col_name in eval_df.columns
    assert f'{response_col_name}_lower' in eval_df.columns
    assert f'{response_col_name}_upper' in eval_df.columns

def test_virtual_best_baseline_recalibrate(stochastic_benchmark_instance):
    vb = VirtualBestBaseline(stochastic_benchmark_instance.get_experiment_parameters())
    new_df = vb.rec_params.sample(2)
    vb.recalibrate(new_df)
    expected_cols = ['resource'] + stochastic_benchmark_instance.parameter_names
    for col in expected_cols:
        assert col in vb.rec_params.columns

# ProjectionExperiment tests
def test_projection_experiment_evaluate(stochastic_benchmark_instance):
    # RESTORED: Setting up training stats with specific naming
    response_metric_col_name = names.param2filename({'Key': 'response', 'Metric': 'median'}, '')
    stochastic_benchmark_instance.training_stats = pd.DataFrame({
        'resource': [1, 2],
        'param1': [10, 20],
        'boots': [1, 1],
        response_metric_col_name: [0.5, 0.6]
    })
    pe = ProjectionExperiment(stochastic_benchmark_instance.get_experiment_parameters(), "TrainingStats")
    params_df, eval_df = pe.evaluate()
    response_col_name = 'response' 
    assert 'resource' in params_df.columns
    assert response_col_name in eval_df.columns
    assert f'{response_col_name}_lower' in eval_df.columns
    assert f'{response_col_name}_upper' in eval_df.columns

def test_projection_experiment_evaluate_monotone(stochastic_benchmark_instance):
    # RESTORED: Detailed setup with complex names
    response_metric_col_name = names.param2filename({'Key': 'response', 'Metric': 'median'}, '')
    response_metric_lower_col_name = names.param2filename({'Key': 'response', 'Metric': 'median', 'ConfInt': 'lower'}, '')
    response_metric_upper_col_name = names.param2filename({'Key': 'response', 'Metric': 'median', 'ConfInt': 'upper'}, '')
    
    stochastic_benchmark_instance.training_stats = pd.DataFrame({
        'resource': [1, 2],
        'param1': [10, 20],
        'boots': [1, 1],
        response_metric_col_name: [0.5, 0.6]
    })
    stochastic_benchmark_instance.testing_stats = pd.DataFrame({
        'resource': [1, 2],
        'param1': [10, 20],
        response_metric_col_name: [0.5, 0.6],
        response_metric_lower_col_name: [0.4, 0.5],
        response_metric_upper_col_name: [0.6, 0.7]
    })
    pe = ProjectionExperiment(stochastic_benchmark_instance.get_experiment_parameters(), "TrainingStats")
    params_df, eval_df = pe.evaluate_monotone()
    response_col_name = 'response'
    diffs = eval_df[response_col_name].diff().fillna(0)
    assert all(diffs >= 0) or all(diffs <= 0)

def test_projection_experiment_recipe_path(stochastic_benchmark_instance):
    response_metric_col_name = names.param2filename({'Key': 'response', 'Metric': 'median'}, '')
    stochastic_benchmark_instance.training_stats = pd.DataFrame({
        'resource': [1, 2],
        'param1': [10, 20],
        'boots': [1, 1],
        response_metric_col_name: [0.5, 0.6]
    })
    pe = ProjectionExperiment(stochastic_benchmark_instance.get_experiment_parameters(), "TrainingStats")
    pe.set_rec_path()
    assert pe.rec_path.endswith(".pkl")
