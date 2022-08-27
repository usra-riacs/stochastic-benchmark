import numpy as np
import names
import matplotlib.pyplot as plt
from scipy.special import erfinv
EPSILON = 1e-10


class SuccessMetrics:
    def __init__(self, shared_args):
        self.shared_args = shared_args #confidence level, best_value, random_values


class Response(SuccessMetrics):
    def __init__(self, shared_args, metric_args):
        SuccessMetrics.__init__(self, shared_args)
        self.name='Response'
        self.opt_sense = metric_args['opt_sense']
        
    def evaluate(self, bs_df, responses, resources):
        if self.opt_sense == -1:  # Minimization
            response_dist = np.apply_along_axis(
                func1d=np.min, axis=0, arr=responses)
        else:  # Maximization
            response_dist = np.apply_along_axis(
                func1d=np.max, axis=0, arr=responses)
        plt.hist(response_dist)
        plt.title('Histogram of samples')
        key = self.name
        basename = names.param2filename({'Key': key}, '')
        CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
        CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')
        
        confidence_level = self.shared_args['confidence_level']
        mean_val = np.mean(response_dist)
        std_dev = np.nanstd(response_dist)
        fact = erfinv(confidence_level / 100.) * np.sqrt(2.)

        bs_df[basename] = [mean_val]
        bs_df[CIlower] = mean_val - fact*std_dev
        bs_df[CIupper] = mean_val + fact*std_dev
        
class PerfRatio(SuccessMetrics):
    def __init__(self, shared_args, metric_args):
        SuccessMetrics.__init__(self, shared_args)
        self.name = 'PerfRatio'
        self.opt_sense = 1
#         self.depends = [Response()]
    
    def evaluate(self, bs_df, responses, resources):
        key = self.name
        basename = names.param2filename({'Key': key}, '')
        CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
        CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')
        
        random_value = self.shared_args['random_value']
        best_value = self.shared_args['best_value']
        
        bs_df[basename] = (random_value - bs_df[names.param2filename({'Key': 'Response'}, '')])\
            / (random_value - best_value)
        bs_df[CIlower] = (random_value - bs_df[names.param2filename({'Key': 'Response', 'ConfInt': 'upper'}, '')]) \
            / (random_value - best_value)
        bs_df[CIupper] = (random_value - bs_df[names.param2filename({'Key': 'Response', 'ConfInt': 'lower'}, '')])\
            / (random_value - best_value)

class InvPerfRatio(SuccessMetrics):
    def __init__(self, shared_args, metric_args):
        SuccessMetrics.__init__(self, shared_args)
        self.name = 'InvPerfRatio'
        self.opt_sense = -1
#         self.depends = [Response()]
    
    def evaluate(self, bs_df, responses, resources):
        key = self.name
        basename = names.param2filename({'Key': key}, '')
        CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
        CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')
        
        random_value = self.shared_args['random_value']
        best_value = self.shared_args['best_value']
        
        bs_df[basename] = 1 - (random_value- bs_df[names.param2filename({'Key': 'Response'}, '')]) / \
            (random_value - best_value) + EPSILON
        bs_df[CIlower] = 1 - (random_value - bs_df[names.param2filename({'Key': 'Response', 'ConfInt': 'lower'}, '')])\
            / (random_value - best_value) + EPSILON
        bs_df[CIupper] = 1 - (random_value - bs_df[names.param2filename({'Key': 'Response', 'ConfInt': 'upper'}, '')])\
            / (random_value - best_value) + EPSILON
        
class SuccessProb(SuccessMetrics):
    def __init__(self, shared_args, metric_args):
        SuccessMetrics.__init__(self, shared_args)
        self.name = 'SuccProb'
        self.opt_sense = 1 #This is opt sense of SuccessProb
        self.args = metric_args #gap to count as success, response_dir is if the underlying col should be max or min
        
    def evaluate(self, bs_df, responses, resources):
        key = self.name
        basename = names.param2filename({'Key': key}, '')
        CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
        CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')
        
        downsample = responses.shape[0]
        random_value = self.shared_args['random_value']
        best_value = self.shared_args['best_value']
        confidence_level = self.shared_args['confidence_level']
        
        if self.shared_args['response_dir'] == -1:
            success_thresh = random_value - \
            (1.0 - self.args['gap']/100.0)*(random_value - best_value)
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x < success_thresh)/downsample, axis=0, arr=responses)
        
        else:  # Maximization
            success_thresh = (1.0 - self.args['gap']/100.0) * \
                (best_value - random_value) - random_value
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x > success_thresh)/downsample, axis=0, arr=responses)
            
        key = self.name
        basename = names.param2filename({'Key': key}, '')
        CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
        CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')

        bs_df[basename] = np.mean(success_prob_dist)
        bs_df[CIlower] = np.nanpercentile(
            success_prob_dist, 50 - confidence_level/2)
        bs_df[CIupper] = np.nanpercentile(
            success_prob_dist, 50 + confidence_level/2)

# This one is kind of weird
class Resource(SuccessMetrics):
    def __init__(self, shared_args, metric_args):
        SuccessMetrics.__init__(self, shared_args)
#         self.args = args
        self.opt_sense = -1
        self.name = 'MeanTime'
        
    def evaluate(self, bs_df, responses, resources):
        resource_dist = np.apply_along_axis(
        func1d=np.mean, axis=0, arr=resources)

        key = self.name
        basename = names.param2filename({'Key': key}, '')
        CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
        CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')
        
        confidence_level = self.shared_args['confidence_level']

        bs_df[basename] = np.mean(resource_dist)
        bs_df[CIlower] = np.nanpercentile(
            resource_dist, 50 - confidence_level/2)
        bs_df[CIupper] = np.nanpercentile(
            resource_dist, 50 + confidence_level/2)

class RTT(SuccessMetrics):
    def __init__(self, shared_args, metric_args):
        SuccessMetrics.__init__(self, shared_args)
        self.name = 'RTT'
        self.args  = metric_args #fail_value, RTT_factor
        self.opt_set = -1
        self.evaluate_vectorized = np.vectorize(self.evaluate_single, excluded=(2, 3))
            
    def evaluate(self, bs_df, responses, resources):
        key = self.name
        basename = names.param2filename({'Key': key}, '')
        CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
        CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')
        
        downsample = responses.shape[0]
        random_value = self.shared_args['random_value']
        best_value = self.shared_args['best_value']
        confidence_level = self.shared_args['confidence_level']
        
        if self.shared_args['response_dir'] == -1:
            success_thresh = random_value - \
            (1.0 - self.args['gap']/100.0)*(random_value - best_value)
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x < success_thresh)/downsample, axis=0, arr=responses)
        
        else:  # Maximization
            success_thresh = (1.0 - self.args['gap']/100.0) * \
                (best_value - random_value) - random_value
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x > success_thresh)/downsample, axis=0, arr=responses)
#         print(len(success_prob_dist))
#         print(self.args['RTT_factor'])
        rtt_dist = self.evaluate_vectorized(
            success_prob_dist, scale=self.args['RTT_factor'])
        # Question: should we scale the RTT with the number of bootstrapping we do, intuition says we don't need to
        rtt = np.mean(rtt_dist)

        bs_df[basename] = rtt
        if np.isinf(rtt) or np.isnan(rtt) or rtt == self.args['fail_value']:
            bs_df[CIlower] = self.args['fail_value']
            bs_df[CIupper] = self.args['fail_value']
        else:
            # rtt_conf_interval = computeRTT_vectorized(
            #     success_prob_conf_interval, s=0.99, scale=1e-6*df_default_samples['runtime (us)'].sum())
            bs_df[CIlower] = np.nanpercentile(
                rtt_dist, 50-confidence_level/2)
            bs_df[CIupper] = np.nanpercentile(
                rtt_dist, 50+confidence_level/2)
    
    def evaluate_single(self, success_probability, scale=1.0, size = 1000):
        if success_probability == 0:
            return self.args['fail_value']
        elif success_probability == 1:
            # Consider continuous RTT and RTT scaled by assuming success_probability=1 as success_probability=1/size*(1-1/10)
            return scale*np.log(1.0 - self.args['s']) / np.log(1 - (1 - 1/10)/size)
        else:
            return scale*np.log(1.0 - self.args['s']) / np.log(1 - success_probability)

