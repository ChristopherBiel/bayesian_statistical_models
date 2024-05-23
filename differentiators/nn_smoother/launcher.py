from differentiators.utils import generate_base_command, generate_run_commands, dict_permutations
import differentiators.nn_smoother.exp as exp

general_configs = {
    'project_name': ['BNNSmootherSweep_240523'],
    'seed': [0, 1],
    'num_traj': [12],
    'sample_points': [64],
    'smoother_particles': [16],
    'dyn_particles': [12, 16],
    'smoother_training_steps': [4000, 8000],
    'dyn_training_steps': [4000, 16000, 64000],
    'smoother_weight_decay': [2e-4],
    'dyn_weight_decay': [8e-4, 2e-4, 8e-5, 2e-5],
    'smoother_type': ['DeterministicEnsemble'],
    'dyn_type': ['DeterministicEnsemble', 'DeterministicFSVGDEnsemble'],
    'logging_mode_wandb': [2],
    'x_src': ['smoother', 'data'],
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
                          duration='23:59:00',
                          prompt=True,
                          mem = 4 * 1028)

if __name__ == '__main__':
    main()    
