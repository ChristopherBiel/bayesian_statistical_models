from differentiators.utils import generate_base_command, generate_run_commands, dict_permutations
import differentiators.exp as exp

general_configs = {
    'project_name': 'BNNSmootherAndDynamics_240513',
    'seed': [0,1],
    'num_traj': [12],
    'noise_level': [None],
    'sample_points': [64],
    'smoother_features': [(128, 64)],
    'dyn_features': [(128, 128, 64)],
    'smoother_particles': [12],
    'dyn_particles': [12],
    'smoother_train_steps': [1000, 4000],
    'dyn_train_steps': [1000, 4000, 16000],
    'smoother_weight_decay': [1e-4 ,3e-4],
    'dyn_weight_decay': [1e-4, 3e-4],
    'smoother_type': ['DeterministicEnsemble', 'ProbabilisticFSVGDEnsemble', 'DeterministicFSVGDEnsemble'],
    'dyn_type': ['DeterministicEnsemble', 'ProbabilisticFSVGDEnsemble', 'DeterministicFSVGDEnsemble'],
    'logging_mode_wandb': [1],
}

def main():
    command_list = []
    flags_combinations = dict_permutations(general_configs)
    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)
    
    # submit the jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          mode='euler',
                          duration='3:59:00',
                          prompt=True,
                          mem = 4 * 1028)

if __name__ == '__main__':
    main()    