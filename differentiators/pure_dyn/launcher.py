from differentiators.utils import generate_base_command, generate_run_commands, dict_permutations
import differentiators.pure_dyn.pure_dyn as exp

general_configs = {
    'project_name': ['BNNSmootherSweep_240523'],
    'seed': [0, 1, 2],
    'num_traj': [12],
    'sample_points': [48, 64, 128],
    'num_traj_train': [1, 2, 4, 12],
    'dyn_particles': [12, 16, 20, 24],
    'dyn_training_steps': [4000, 16000, 24000, 32000, 480000, 64000],
    'dyn_weight_decay': [8e-4, 2e-4, 8e-5, 2e-5],
    'dyn_type': ['DeterministicEnsemble', 'DeterministicFSVGDEnsemble'],
    'logging_mode_wandb': [2],
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
                          num_gpus=0,
                          mode='euler',
                          duration='23:59:00',
                          prompt=True,
                          mem = 4 * 1028)

if __name__ == '__main__':
    main()    
