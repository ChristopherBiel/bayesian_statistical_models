from differentiators.utils import generate_base_command, generate_run_commands, dict_permutations
import differentiators.pure_dyn.pure_dyn as exp

general_configs = {
    'project_name': ['DynModelOnly_Sweep_240524'],
    'seed': [0, 1],
    'num_traj': [12],
    'sample_points': [32, 64],
    'num_traj_train': [1, 2, 12],
    'dyn_features': [32, 64, 128],
    'dyn_particles': [10],
    'dyn_training_steps': [1000, 4000, 16000],
    'dyn_weight_decay': [2e-4, 8e-5],
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
